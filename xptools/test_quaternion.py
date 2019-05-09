from motive_data_helpers import *
import quaternion as q
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
import pdb

def quat_to_matlab(file_q):
    data_q = readfile(file_q, 'stop2')
    quats = data_q[['Rotation'+k for k in ['W','X','Y','Z']]].values
    savemat(file_q[:-3]+'mat', dict(q=quats))

def compare_files(file_e, file_q, description, option):
    data_q = readfile(file_q, 'stop2')

    if option == 'optitrack':
        data_e = readfile(file_e, 'stop2')
        data_e['phi'] = data_e['RotationX']
        data_e['theta'] = data_e['RotationZ']
        data_e['psi'] = -data_e['RotationY']
    elif option == 'matlab':
        data_e = pd.read_csv(file_e, header=None, names=['phi','theta','psi'])
        data_e['Time'] = data_q.Time

    data_q['phi'], data_q['theta'], data_q['psi'] = euler2quat(data_q)

    # Phi
    plt.plot(data_q.Time, data_q.phi)
    plt.plot(data_e.Time, data_e.phi)
    plt.legend(['from_quat', 'from_euler'])
    plt.xlabel('time')
    plt.ylabel('$\phi$')
    plt.title(description['phi'])
    plt.show()

    # Theta
    plt.plot(data_q.Time, data_q.theta)
    plt.plot(data_e.Time, data_e.theta)
    plt.legend(['from_quat', 'from_euler'])
    plt.xlabel('time')
    plt.ylabel('$\\theta$')
    plt.title(description['theta'])
    plt.show()

    # Psi
    plt.plot(data_q.Time, data_q.psi)
    plt.plot(data_e.Time, data_e.psi)
    plt.legend(['from_quat', 'from_euler'])
    plt.xlabel('time')
    plt.ylabel('$\psi$')
    plt.title(description['psi'])
    plt.show()

if __name__ == '__main__':
    # Case 1
    # file_e = os.path.join(os.getcwd(), 'calib_euler/take1_eul.csv')
    file_e = os.path.join(os.getcwd(), 'calib_euler/take1_matlab.csv')
    file_q = os.path.join(os.getcwd(), 'calib_euler/take1_quat.csv')
    description = {
        'phi':'roll: +45, +90, -45, -90',
        'theta':'pith: +45, +90, -45, -90',
        'psi':'yaw: +90, 0, -90, -180, -270, -360'
    }
    compare_files(file_e, file_q, description,option='matlab')


    # case 2
    file_q = os.path.join(os.getcwd(), 'calib_euler/take2_quat.csv')
    # file_e = os.path.join(os.getcwd(), 'calib_euler/take2_eul.csv')
    file_e = os.path.join(os.getcwd(), 'calib_euler/take2_matlab.csv')
    description = {
        'phi': 'roll: 30, 30, 30 (approx)',
        'theta': 'pith:  30, 30, 30 (approx)',
        'psi': 'yaw:  30, -30, -30 (approx)'
    }
    compare_files(file_e, file_q, description, option='matlab')