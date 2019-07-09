import os
from xptools.motive_data_helpers import *
from xptools.smoothing_helpers import hampel_filter, ParallelCrossValidation, FusedLassoFilter
from scipy.stats import mode
import pdb
from xptools.vocab import *
from json import load as jsonload
from pyfme.models import RigidBodyEuler, RigidBodyQuat
from pyfme.aircrafts.aircraft import Aircraft
from pyfme.environment.environment import Environment
from joblib import Parallel, delayed
from cvxpy.error import SolverError
import matplotlib.pyplot as plt
import argparse
import json
import pandas as pd
from scipy.interpolate import interp1d
from pyfme.utils.change_euler_quaternion import change_basis
import os

import matplotlib
matplotlib.use('agg')

## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', default=r'raw_trajs_filtered', help='file where to find raw trajectories')
parser.add_argument('-o', '--output_dir', default=r'camera_ready', help='file where to find processed trajectories')
parser.add_argument('-p', '--params', default='smoothing_options.json',help="Parameter json file with smoothing constants")
parser.add_argument('-m','--meta_dir',default='smoothing_meta',help='File where to save smoothing information')


if __name__ == '__main__':
    # Parse inputs
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    meta_dir = args.meta_dir
    with open(args.params, 'r') as f:
        par = json.load(f)
    actions = par['actions']
    if par['files'] == None:
        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    else:
        files= par['files']
    data = {}
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    ''' ---------- Load Data---------- '''
    for file in files:
        # Read file
        filefull = os.path.join(input_dir, file)
        data[file] = pd.read_csv(filefull)
    
    # if actions['flip']:
    #     for file in files:
    #         if data[file].x_e[:10].mean() > data[file].x_e[-10:].mean():
    #             data[file][['x_e','y_e']] *= -1
    #             temp_quat = data[file][quat].copy()
    #             data[file].q0 = -temp_quat.qz
    #             data[file].qx = -temp_quat.qy
    #             data[file].qy = temp_quat.qx
    #             data[file].qz = temp_quat.q0
    #             # if data[file].q0.mean() < 0:
    #             #     data[file][quat] *= -1

    if actions['smooth']:
        ''' ---------- Remove outliers and replace by NaN ---------- '''
        hampel_params = par['hampel_filter']
        print("Removing outliers with Hampel filter")
        raw_data = {}
        for file in files:
            # Create dataframe for raw values
            raw_data[file] = data[file].copy()
            data[file] = pd.DataFrame(columns=data[file].columns)

            # Interpolate in the case of missing time steps
            dt = raw_data[file].time.diff().median()
            N = np.round((raw_data[file].time.iloc[-1] - raw_data[file].time.iloc[0])/dt)
            data[file].time = dt * np.arange(N)
            for field in pos+quat:
                f = interp1d(raw_data[file].time, raw_data[file][field])
                data[file][field] = f(data[file].time)

            # Get rid of outliers with hampel filter
            data[file][pos] = hampel_filter(data[file][pos], **hampel_params['hampel_params_pos']).values
            data[file][quat] = hampel_filter(data[file][quat], **hampel_params['hampel_params_quat']).values


        ''' ---------- Smooth and complete missing data with Fused Lasso ---------- '''
        print("Smoothing and completing missing data with Fused Lasso")
        smoothness = par['fused_lasso_filter']['smoothness']
        # if smoothness = -1 then we need to perform cross-validation
        if smoothness == -1:
            print("Cross Validation required for Fused Lasso smoothing parameter.")
            smoothness = {}
            cv = par['fused_lasso_filter']['cross_validation_params']
            # Run cross validation for positions
            s_table = np.logspace(*cv['pos_ini_end_n']) if cv['scale'] == 'log' else np.linspace(*cv['pos_ini_end_n'])
            smoothness, errors_table = ParallelCrossValidation(pos, s_table, par, cv, files, meta_dir, data, smoothness)

            # Run cross validation for quaternions
            s_table = np.logspace(*cv['quat_ini_end_n']) if cv['scale'] == 'log' else np.linspace(*cv['quat_ini_end_n'])
            smoothness, errors_table = ParallelCrossValidation(quat, s_table, par, cv, files, meta_dir, data, smoothness)

        if type(smoothness) == float:
            smoothness = {field: smoothness for field in pos+quat}
            print(smoothness)

        for file in files:
            for field in pos+quat:
                try:
                    data[file][field], status = FusedLassoFilter(data[file][field].values, smoothness[field])
                except (AssertionError, SolverError) as e:
                    print("Filtering did not converge for field {} of file {} with smoothness {}. Error message: {}"\
                    .format(field, file, smoothness[field], e))

            # Renormalize quaternions after filtering
            #TODO: improve FusedLassoFilterQuat in motive_data_helpers
            nrm = np.linalg.norm(data[file][quat].values, axis=1)
            for field in quat:
                data[file][field] /= nrm

    ''' ---------- Correct pivot point - CG offset ---------- '''
    aircraft_file = par['aircraft']
    with open(aircraft_file, 'r') as f:
            aircraft_caract = jsonload(f)
    for f in files:
        delta_body = - np.array(aircraft_caract['CGlocation']) + np.array(aircraft_caract['pivot_point_location'])
        delta = rotate_vector(np.array(delta_body),data[file][quat])
        data[file][pos] -= delta

    ''' ---------- Get Euler angles ---------- '''
    for f in files:
        quat2euler(data[f])
        unwrap_euler(data[f])
        for k in eul:
            data[f][k + '_deg'] = data[f][k] * 180 / np.pi

    if actions['compute_derivatives']:
        ''' ---------- Differentiate to get velocities and derivative of quaternions in Earth Frame ---------- '''
        for f in files:
            dt = mode(np.diff(data[f].time.values))[0][0]
            differentiate(data[f], dt=dt, fields=pos+quat)
            differentiate(data[f], dt=dt, fields=dpos+dquat)

    if actions['compute_body_fr']:

        ''' ---------- Invert Dynamics ---------- '''
        # Create aircraft and sys of equations
        a = Aircraft()
        with open(aircraft_file, 'r') as f:
            aircraft_caract = jsonload(f)
        # a.mass = aircraft_caract['mass']
        a.mass = aircraft_caract['inertia']['mass']
        # a.inertia = np.array(aircraft_caract['inertia'])
        a.inertia = np.array(aircraft_caract['inertia']['inertia'])
        try:
            # a.chord = (aircraft_caract['Wing']['root chord'] + aircraft_caract['Wing']['tip chord']) / 2
            # a.span = aircraft_caract['Wing']['span']
            a.chord = aircraft_caract['geometry']['chord']
            a.span = aircraft_caract['geometry']['span']
            a.Sw = aircraft_caract['geometry']['Sw']
            # a.Sw = a.chord * a.span
        except:
            a.chord = aircraft_caract['chord']
            a.span = aircraft_caract['span']
            a.Sw = aircraft_caract['Sw']
        if par['allEuler']:
            sys = RigidBodyEuler(aircraft=a, environment=Environment())
            attitude_type = 'eul'
        else:
            sys = RigidBodyQuat(aircraft=a, environment=Environment())
            attitude_type = 'quat'

        # Invert dynamics
        for f in files:
            for k in frc+mom+vel+rot+aerof+aerocf+aerocm+perf:
                data[f][k] = float('nan')
            data[f] = invert_dynamics(data[f], sys, dt, type=attitude_type)


    ''' ---------- Save all data ---------- '''
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for file in files:
        data[file] = data[file].iloc[2:-2]
        filefull = os.path.join(output_dir, file)
        data[file].to_csv(filefull, index=False)


