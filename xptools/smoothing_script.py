import os
from motive_data_helpers import *
from smoothing_helpers import *
from scipy.stats import mode
import pdb
from vocab import *
from json import load as jsonload
from pyfme.models import RigidBodyEuler, RigidBodyQuat
from pyfme.aircrafts.aircraft import Aircraft
from pyfme.environment.environment import Environment
from joblib import Parallel, delayed
from cvxpy.error import SolverError
import matplotlib.pyplot as plt
import argparse
import json

import matplotlib
matplotlib.use('agg')

## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', default=r'raw_trajs', help='file where to find raw trajectories')
parser.add_argument('-o', '--output_dir', default=r'camera_ready', help='file where to find processed trajectories')
parser.add_argument('-p', '--params', default='smoothing_options.json',help="Parameter json file with smoothing constants")


if __name__ == '__main__':
    # Parse inputs
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    with open(args.params, 'r') as f:
        par = json.load(f)
    actions = par['actions']
    if par['files'] == None:
        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    else:
        files= par['files']
    data = {}

    ''' ---------- Load Data---------- '''
    for file in files:
        # Read file
        filefull = os.path.join(input_dir, file)
        data[file] = pd.read_csv(filefull)


    if actions['smooth']:
        ''' ---------- Remove outliers and replace by NaN ---------- '''
        hampel_params = par['hampel_filter']
        print("Removing outliers with Hampel filter")
        for file in files:
            # Create data columns for raw values
            for sufx in pos+quat:
                data[file][sufx + '_raw'] = data[file][sufx]

            # Get rid of outliers with hampel filter
            data[file][pos] = hampel_filter(data[file][pos], **hampel_params['hampel_params_pos']).values
            data[file][quat] = hampel_filter(data[file][quat], **hampel_params['hampel_params_quat']).values


        ''' ---------- Smooth and complete missing data with Fused Lasso ---------- '''
        print("Smoothing and completing missing data with Fused Lasso")

        smoothness = par['fused_lasso_filter']['smoothness']
        # if smoothness = -1 then we need to perform cross-validation
        if smoothness == -1:
            cv = par['fused_lasso_filter']['cross_validation_params']
            s_table = np.logspace(*cv['ini_end_n']) if cv['scale'] == 'log' else np.linspace(*cv['ini_end_n'])
            print("Cross Validation required for Fused Lasso smoothing parameter.")
            smoothness = {}
            for field in pos+quat:
                errors = np.array(Parallel(n_jobs=5)(delayed(
                    FusedLassoCrossValidation)(data[file][field], s_table, cv['number_random_folds']) for file in files))
                # FusedLassoCrossValidation(data['trajectory77.csv'][field], s_table, cv['number_random_folds'])
                errors = np.sum(errors, axis=0)
                plt.loglog(s_table, errors)
                plt.legend(pos+quat)
                plt.grid(True)
                plt.xlabel('smoothness parameter')
                plt.ylabel('Summed Cv error on all trajectories')
                plt.savefig('crossvalid.png')
                errors[np.isnan(errors)] = 1e10
                smoothness[field] = s_table[np.argmin(errors)]
        print(smoothness)

        if type(smoothness) == float:
            smoothness = {field: smoothness for field in pos+quat}

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
        aircraft_file = par['aircraft']

        ''' ---------- Invert Dynamics ---------- '''
        # Create aircraft and sys of equations
        a = Aircraft()
        with open(aircraft_file, 'r') as f:
            aircraft_caract = jsonload(f)
        a.mass = aircraft_caract['mass']
        a.inertia = np.array(aircraft_caract['inertia'])
        try:
            a.chord = (aircraft_caract['Wing']['root chord'] + aircraft_caract['Wing']['tip chord']) / 2
            a.span = aircraft_caract['Wing']['span']
            a.Sw = a.chord * a.span
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
    for file in files:
        data[file] = data[file].iloc[2:-2]
        filefull = os.path.join(output_dir, file)
        data[file].to_csv(filefull, index=False)


