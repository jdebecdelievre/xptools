import pandas as pd
import os
from asi.preprocessing.pandas_basic import *
import numpy as np
import argparse
from motive_data_helpers import *
from scipy.stats import mode
import pdb
from vocab import *
from json import load as jsonload

from pyfme.models import RigidBodyEuler, RigidBodyQuat
from pyfme.aircrafts.aircraft import Aircraft
from pyfme.environment.environment import Environment
from pyfme.environment.atmosphere import SeaLevel
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind
from pyfme.utils.anemometry import calculate_alpha_beta_TAS
from pyfme.utils.coordinates import body2wind
import cvxpy
from scipy.signal import lfilter


## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', default=r'raw_trajs', help='file where to find raw trajectories')
parser.add_argument('-o', '--output_dir', default=r'camera_ready', help='file where to find processed trajectories')
parser.add_argument('-a', '--aircraft', default=r'aircraft.json', help='Aircraft to be used')
parser.add_argument('-eul', action='store_true', help='Use Euler angles to get body frame variables. Default is to use quaternion instead')
parser.add_argument('-s','--smoothness', help="Smoothness coefficient for fused lasso. Lower coefficient for smoother fit.  0 for cross-validation")

## PARAMETERS
hampel_default = dict(k=7, t0=3, sigma_noise=0)
learn_columns = time+vel+rot+frc+mom+pos+[p+'_filt' for p in pos]
hampel_params_eul = dict(k=9, t0=3, sigma_noise=0.005)
hampel_params_pos = dict(k=7, t0=3, sigma_noise=0.01)
min_traj_size = 10

def differentiate(data, dt, fields=pos_eul, filt=True):
    # Perform second order central differences, except at edges (2nd order bwd or fwd)

    # Treat missing arguments
    if fields is None:
        fields = data.columns

    # Loop on groups and on values to differentiate
    for sufx in fields:
        # Use filtered data or not
        sufx_filt = sufx+'_filt' if filt else sufx    
        data[sufx + '_dot'] = np.gradient(data[sufx_filt].values, dt, axis=0, edge_order=2)
    return data

def invert_dynamics(data, sys, dt, type):
    if type=='eul':
        forces, moments, state, state_dot = \
            sys.inverse_dynamics(dt, 
                             data[fpos].values, 
                             data[feul].values, 
                             data[dpos].values, 
                             data[deul].values)
    elif type == 'quat':
        forces, moments, state, state_dot = \
            sys.inverse_dynamics(dt, 
                             data[fpos].values, 
                             data[fquat].values, 
                             data[dpos].values, 
                             data[dquat].values)            
    data[frc] = forces
    data[mom] = moments
    data[vel] = state.velocity
    data[rot] = state.omega

    # Add wind variables
    to_wind_frame(data)
    performance_params(data)

    # Add non-dimensional forces and moments
    conditions = sys.environment.calculate_aero_conditions(state)
    for i in range(3):
        data[aerocf[i]] = data[aerof[i]]/(1/2 * conditions.q_inf * sys.aircraft.Sw)
        data[aerocm[i]] = data[mom[i]]/(1/2 * conditions.q_inf * sys.aircraft.Sw * sys.aircraft.chord)
    return data
   

def to_wind_frame(data):
    alpha, beta, TAS = calculate_alpha_beta_TAS(data[vel].values)
    for i, ind in enumerate(data.index):
        data.loc[ind, aerof] = body2wind(data.loc[ind, frc], alpha[i], beta[i]) * np.array([-1,1,-1])
    return data


def performance_params(data):
    data['L_D'] = data.L / data.D
    return data


def hampel_filtering(data, hampel_params=hampel_default, fields=pos_eul):
    # Removing the Setting WithCopyWarning
    data = data.copy()


    # Treat missing arguments
    if fields is None:
        fields = data.columns

    # Create data columns for filtered values
    fields_filt = [sufx + '_filt' for sufx in fields]
    for sufx in fields_filt:
        data[sufx] = float('nan')

    # Get rid of outliers with hampel filter
    data[fields_filt]= hampel_filter(data[fields], **hampel_params).values

    return data


def linear_filter(aa, bb, X, padlen=None):  
    if padlen is None:
        padlen = 3 * max(len(aa), len(bb)) # default value in filtfilt
    try:
        Xf = filtfilt(aa, bb, X, axis=0, padlen=padlen)
    except ValueError:
        Xf = linear_filter(aa, bb, X, padlen-1)
    return Xf


def find_nonNaN(data):
    vals = dict()
    n_beginning = 0
    n_groups = 0
    recording = False
    for n in data.index:
        if recording and data.loc[n].isna().any():
            if (n - 1) - n_beginning > min_traj_size:
                n_groups += 1
                vals[n_groups] = [n_beginning, n - 1]
            recording = False
        elif not recording and not data.loc[n].isna().any():
            n_beginning = n
            recording = True
    if recording:
        n_groups += 1
        vals[n_groups] = (n_beginning, n)
    return vals

def unwrap_euler(data):
    data['phi'] = np.unwrap(data.phi)
    data['psi'] = np.unwrap(data.psi)
    data['theta'] = np.unwrap(data.theta * 2) / 2


if __name__ == '__main__':
    # Parse inputs
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir


    # Add relevant columns
    if not args.eul:
        learn_columns += quat+[q+'_filt' for q in quat]
    else:
        learn_columns += eul+[e+'_filt' for e in eul]

    DATA = []
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv') ]

    # Create aircraft and sys of equations
    a = Aircraft()
    with open(args.aircraft, 'r') as f:
        aircraft_caract = jsonload(f)
    a.mass = aircraft_caract['mass']
    a.inertia = np.array(aircraft_caract['inertia'])
    try:
        a.chord = (aircraft_caract['Wing']['root chord'] + aircraft_caract['Wing']['tip chord'])/2
        a.span = aircraft_caract['Wing']['span']
        a.Sw = a.chord * a.span
    except:
        a.chord = aircraft_caract['chord']
        a.span = aircraft_caract['span']
        a.Sw = aircraft_caract['Sw']
    if args.eul:
        sys = RigidBodyEuler(aircraft=a, environment=Environment())
        attitude_type = 'eul'    
    else:
        sys = RigidBodyQuat(aircraft=a, environment=Environment())     
        attitude_type = 'quat'     

    ## Loop on files
    for file in files:
        # Read file
        filefull = os.path.join(input_dir, file)
        raw_data = pd.read_csv(filefull)
        print(file)

        ''' ---------- Find groups of non-Nan data and remove outliers ---------- '''
        vals = find_nonNaN(raw_data)
        traj_data = []
        for n in list(vals.keys()):
            data = raw_data.loc[vals[n][0]:vals[n][1]].copy()

            # Filter data where there is no NaN
            data = hampel_filtering(data, hampel_params=hampel_params_pos, fields=pos)
            data = hampel_filtering(data, hampel_params=hampel_params_eul, fields=quat)

        ''' ---------- Smooth and complete missing data with Fused Lasso ---------- '''
        for field in pos+quat:
            data[field] = FusedLassoFilter(data[field].value, args.smoothness)

        # Renormalize quaternions after filtering


            # get Euler angles
            quat2euler(data)
            unwrap_euler(data)
            for k in eul:
                data[k + '_deg'] = data[k] * 180 / np.pi

            # Differentiate to have velocities in earth frame
            dt = mode(np.diff(data.time.values))[0][0]
            differentiate(data, dt=dt, fields=pos+quat)

            # Invert dynamics
            for k in frc+mom+vel+rot+aerof+aerocf+aerocm+perf:
                data[k] = float('nan')
            data = invert_dynamics(data, sys, dt, type=attitude_type)

            # Save for later 
            data = data.iloc[2:-2]
            traj_data.append(data)

        # Override csv files
        filefull = os.path.join(output_dir, file)
        traj_data = pd.concat(traj_data)
        traj_data.to_csv(filefull, index=False)

        DATA.append(traj_data[learn_columns])
    DATA = pd.concat(DATA)
    DATA.to_csv('all_data.csv',index=False)


