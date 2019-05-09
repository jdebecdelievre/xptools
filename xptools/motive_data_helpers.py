import pandas as pd
import csv
import numpy as np
from collections import defaultdict
import quaternion
from vocab import *
from pyfme.utils.change_euler_quaternion import quatern2euler, euler2quatern, rotate_vector
from pyfme.utils.coordinates import body2wind

from scipy.signal import filtfilt
from pyfme.utils.anemometry import calculate_alpha_beta_TAS

# PARAMS
hampel_default = dict(k=7, t0=3, sigma_noise=0)

def edges_bool(data, xmin, xmax, ymin, ymax, zmin, zmax):
    bools = (data.x_e > xmin) & (data.x_e < xmax) &\
            (data.y_e > ymin) & (data.y_e < ymax) & \
            (data.z_e < zmax)
    if zmin is not None:
        bools = bools & (data.z_e > zmin)
    return bools


def min_distance(point, walls_def):
# Compute the distance of each edge point to the bounds
# walls_def = {wall_name: (point on plane, normal vector)}
    min_dist = 1e7
    closest_wall = ''
    for wall in walls_def:  
        dist = np.dot(point - walls_def[wall][0],walls_def[wall][1])
        if dist < min_dist:
            closest_wall = wall
            min_dist = dist
    return closest_wall, min_dist
            

def print_marker_names(file):
    csvfile =  open(file, 'r')
    reader = csv.reader(csvfile)
    for i in range(4):
        marker_names = next(reader)
    for m in set(marker_names):
        if len(m)>0:
            print(m)
    csvfile.close()


# Open Motive File and extract data
def readMotiveFile(file, rigid_body, all_markers = False, position_only=False):
    
    # open file
    csvfile =  open(file, 'r')
    reader = csv.reader(csvfile)
    
    # skip header to rigid body names
    for i in range(4):
        column_title = next(reader)
    
    # add rigid body names of interest in a dict(marker_name:marker_data)
    columns = defaultdict(lambda: [1])
    for i in range(len(column_title)):
        if (column_title[i] == rigid_body):
            columns[rigid_body].append(i)
        elif (rigid_body in column_title[i] and all_markers):
            columns[column_title[i].split(':')[1]].append(i)
    
    # get column names and close file
    next(reader)
    motion = next(reader)
    axis = next(reader)
    csvfile.close()
    
    # get data with pd.read
    data = {}
    for marker_name in columns:
        if position_only:
            cols = [1]
            names = ['Time']
            for c in columns[marker_name]:
                if 'Position' in motion[c]:
                    names.append(motion[c]+axis[c])
                    cols.append(c)
            columns[marker_name] = cols
        else:        
            names = [motion[c]+axis[c] for c in columns[marker_name]]
            names[0] = 'Time'
        _dat = pd.read_csv(file, skiprows=7, usecols=columns[marker_name], header=None, names=names)
        
        # renames columns according to raw2clean in vocab.py and swap axis to z-up
        _dat.rename(columns=raw2clean,inplace=True)
        data[marker_name] = swap_axis(_dat)

    # return a single data frame for the pivot point, or a dict with every marker
    if all_markers:
        return data
    else:
        return data[rigid_body]

# Read csv files created by ROS
def readRosFile(file, position_only=False):
    names = ['time', 'x_e', 'y_e', 'z_e'] if position_only else ['time', 'x_e', 'y_e', 'z_e', 'qx', 'qy', 'qz', 'q0'] 
    cols = [2,4,5,6] if position_only else [2,4,5,6,7,8,9,10]
    data = pd.read_csv(file, usecols=cols, names=names, skiprows=1)
    data[['y_e','z_e']] *= -1
    if not position_only:
        data[['qx','q0']] *= -1
    ini_time = data.time[0] * 1.0 * 1e-9
    data.time = data.time * 1.0 * 1e-9 - ini_time
    return data, ini_time


def quat2euler(data, input_col=quat, output_col=eul):
    ''' 
    In place method that creates the phi, theta, psi columns in the dataframe from the 
    4 quaternions columns: q0, qx, qy, qz
    Parameters
    ----------
    data : pandas dataframe
    input_col : list of 4 strings giving the column names for quaternions q0-qx-qy-qz order
    output_col : list of 3 strings giving the column names for euler angles in phi-theta-psi order
    Returns
    -------
    data
   '''
    # import pdb
    # pdb.set_trace()
    euler_angles = quatern2euler(data[input_col].values)
    for i in range(3):
        data[output_col[i]] = euler_angles.T[i] 
    return data


def euler2quat(data, input_col=eul, output_col=quat, q0_sign=None):
    ''' 
    In place method that creates the quaterninons columns q0, qx, qy, qz
    in the dataframe from the 3 euler angles columns: phi, theta, psi
    Parameters
    ----------
    data : pandas dataframe
    input_col : list of 3 strings giving the column names for euler angles in phi-theta-psi order
    output_col : list of 4 strings giving the column names for quaternions q0-qx-qy-qz order
    q0_sign : one float or an array of size len(data) with the desired sign of the first quaternion

    Returns
    -------
    data
   '''
    quaternions = euler2quatern(data[input_col].values)
    if q0_sign is not None:
        quaternions = (2 * (np.sign(quaternions[:,0]) == q0_sign) - 1)[:, np.newaxis] * quaternions
    for i in range(4):
        data[output_col[i]] = quaternions.T[i] 
    return data


def compute_body_frame_variables(data):
    # process Euler Angles
    cos_phi = np.cos(data['phi'])
    sin_phi = np.sin(data['phi'])
    cos_theta = np.cos(data['theta'])
    sin_theta = np.sin(data['theta'])
    cos_psi = np.cos(data['psi'])
    sin_psi = np.sin(data['psi'])

    # add body frame velocity
    data['u'] = cos_theta * cos_psi * data['x_e_dot'] \
              + cos_theta * sin_psi * data['y_e_dot'] \
              - sin_theta * data['z_e_dot']
    data['v'] = (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) * data['x_e_dot'] \
              + (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) * data['y_e_dot'] \
              +  sin_phi * cos_theta * data['z_e_dot']
    data['w'] = (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) * data['x_e_dot'] \
              + (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) * data['y_e_dot'] \
              +  cos_phi * cos_theta * data['z_e_dot']

    # add body frame rotation rates
    data['p'] = data['phi_dot'] - sin_theta * data['psi_dot']
    data['q'] = cos_phi * data['theta_dot'] + sin_phi * cos_theta * data['psi_dot']
    data['r'] =-sin_phi * data['theta_dot'] + cos_phi * cos_theta * data['psi_dot']

def compute_wind_variables(data):
    data['alpha'] = np.arctan(data['w'], data['u'])
    V = np.sqrt(data['u']**2 + data['v']**2 + data['w']**2)
    data['beta'] = np.arcsin(data['v'], V)


# Swap axis to z down
def swap_axis(df):
    '''
    Warning: this function modifies df in place
    '''
    # Switch y and z axis for position
    z = -df.y_e.copy()
    df.y_e = df.z_e
    df.z_e = z
    # Rotate all quaternions around x axis
    if 'q0' in df.columns:
        rot = np.quaternion(np.cos(np.pi/4), -np.sin(np.pi/4),0,0)
        Q = quaternion.as_quat_array(df[quat].values)
        Q = rot * Q * rot.conjugate()
        df[quat] = quaternion.as_float_array(Q)
    return df

def differentiate(data, dt, fields=pos_eul):
    # Perform second order central differences, except at edges (2nd order bwd or fwd)

    # Treat missing arguments
    if fields is None:
        fields = data.columns

    # Loop on groups and on values to differentiate
    for sufx in fields:
        # Use filtered data or not
        data[sufx + '_dot'] = np.gradient(data[sufx].values, dt, axis=0, edge_order=2)
    return data


def invert_dynamics(data, sys, dt, type):
    if type == 'eul':
        forces, moments, state, state_dot = \
            sys.inverse_dynamics(dt,
                                 data[pos].values,
                                 data[eul].values,
                                 data[dpos].values,
                                 data[deul].values)
    elif type == 'quat':
        forces, moments, state, state_dot = \
            sys.inverse_dynamics(dt,
                                 data[pos].values,
                                 data[quat].values,
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
        data[aerocf[i]] = data[aerof[i]] / (conditions.q_inf * sys.aircraft.Sw)
        data[aerocm[i]] = data[mom[i]] / (conditions.q_inf * sys.aircraft.Sw * sys.aircraft.chord)
    return data


def to_wind_frame(data):
    alpha, beta, TAS = calculate_alpha_beta_TAS(data[vel].values)
    for i, ind in enumerate(data.index):
        data.loc[ind, aerof] = body2wind(data.loc[ind, frc], alpha[i], beta[i]) * np.array([-1, 1, -1])
    return data


def performance_params(data):
    data['L_D'] = data.L / data.D
    return data


def linear_filter(aa, bb, X, padlen=None):
    if padlen is None:
        padlen = 3 * max(len(aa), len(bb))  # default value in filtfilt
    try:
        Xf = filtfilt(aa, bb, X, axis=0, padlen=padlen)
    except ValueError:
        Xf = linear_filter(aa, bb, X, padlen - 1)
    return Xf


def find_nonNaN(data, min_traj_size):
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