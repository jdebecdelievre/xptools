from xptools.motive_data_helpers import *
import argparse
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from collections import defaultdict
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib
matplotlib.use('agg')
from xptools.vocab import *
from scipy.interpolate import UnivariateSpline
from xptools.inertia_estimation import run_inertia_estimation, mass_cg_estimation

DEFAULT_TITLE = {
    'alpha': 'Angle of attack (deg)',
    'phi': 'Roll angle (deg)',
    'theta': 'Pitch angle (deg)',
    'psi': 'Yaw angle (deg)',
    'attitude': 'Euler angle (deg)',
    'time': 'Time (s)',
    'p': 'p (rad/s)',
    'q': 'q (rad/s)',
    'r': 'r (rad/s)',
    'phat': '$\hat{p}$',
    'qhat': '$\hat{q}$',
    'rhat': '$\hat{r}$',
    'omega': 'rotation rates (rad/s)',
    'quat': 'quaternion',
    'q0': 'q0',
    'qx': 'qx',
    'qy': 'qy',
    'qz': 'qz',
    'velocity': 'velocity (m/s)',
    'u': 'u (m/s)',
    'v': 'v (m/s)',
    'w': 'w (m/s)',
    'x_e': 'x (m)',
    'y_e': 'y (m)',
    'z_e': 'z (m)',
    'position': 'position (m)',
    'earth_velocity' : 'velocity (m/s)'
}

rcParams = {'legend': {'fontsize': 15},
 'axes': {'labelsize': 20, 'titlesize': 20},
 'figure': {'figsize': (16, 8)}}


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--glider_name', default=r'balsa_glider1',
                    help="Name of rigid body in csv file")
parser.add_argument('-a', '--aircraft_file', default=r'aircraft.json',
                    help="Name of json file with measured information about the aircraft")
parser.add_argument('-i', '--input_location', default=r'take.csv',
                    help="if file_type=Motive, CSV file extracted from optitracks. if file_type=ros, folder name where CSV files are present")
parser.add_argument('-marker_names', action='store_true', help='Show available rigid body names')
parser.add_argument('-noplot', action='store_true', help='No plot')
parser.add_argument('-o', '--output_folder', default='raw_trajs')
parser.add_argument('-t','--file_type', default='ros') # ros or motive
parser.add_argument('--metafiles_location', default='.', help='Folder where to save files related to the cleanup process')



# PARAMETERS
zdelta = -0.3  # minus minimum height any airplane marker can reach

def glider_preprocess(glider_name='balsa_glider', input_location='take.csv', marker_names=False, aircraft_file='aircraft.json',
                        noplot=False, output_folder='raw_trajs', file_type='ros', metafiles_location='.'):
    '''
    glider_name: Name of rigid body in csv file
    marker_names: Show available rigid body names
    file_type: f file_type=Motive, CSV file extracted from optitracks. if file_type=ros, folder name where CSV files are present
    metafiles_location: Folder where to save files related to the cleanup processaircraft_file
    aircraft_file: Name of json file with measured information about the aircraft
    '''
    # Parse inputs
    inp_loc = input_location
    glider_name = glider_name
    if not os.path.isdir(output_folder):
        print('Creating neaw folder: '+output_folder)
        os.makedirs(output_folder, exist_ok=True)
    if not os.path.isdir(metafiles_location):
        os.makedirs(metafiles_location, exist_ok=True)
        
        
    # Print names of markers
    if marker_names:
        print_marker_names(inp_loc)
        exit()

    ''' ---------- Load data ---------- '''
    print('Loading data')
    if file_type == 'motive':
        data = readMotiveFile(inp_loc, glider_name)
        corner1 = readMotiveFile(inp_loc, "corner1", position_only=True)
        corner2 = readMotiveFile(inp_loc, "corner2", position_only=True)
        nostop1 = False
        try:
            stop1 = readMotiveFile(inp_loc, "stop1", position_only=True)
        except:
            print('stop1 not found.')
            nostop1 = True
        nostop2=False
        try:
            stop2 = readMotiveFile(inp_loc, "stop2", position_only=True)
        except:
            print('stop2 not found.')
            nostop2 = True
        if nostop1 and not nostop2:
            stop1 = stop2
        if nostop2 and not nostop1:
            stop2 = stop1

    elif file_type == 'ros':
        data, ini_time = readRosFile(os.path.join(inp_loc, glider_name + '.csv'))
        corner1, _ = readRosFile(os.path.join(inp_loc, "corner1.csv"), position_only=True)
        corner2, _ = readRosFile(os.path.join(inp_loc, "corner2.csv"), position_only=True)
        stop1, ini_s1 = readRosFile(os.path.join(inp_loc, "stop1.csv"), position_only=True)
        stop2, ini_s2 = readRosFile(os.path.join(inp_loc, "stop2.csv"), position_only=True)
    else:
        raise AttributeError

    ''' ---------- Load individual markers on airplane ---------- '''
    if file_type == 'motive':
        indivi_markers = readMotiveFile(inp_loc, glider_name, all_markers=True)
    elif file_type == 'ros':
        indivi_markers, _ = readRosFile(os.path.join(inp_loc, glider_name + '.csv'))
        
    ''' ---------- Show overview in html ---------- '''
    # if not noplot:
    #     print('Showing flight overview in raw_3d.html')
    #     trace=go.Scatter3d(x=data.x_e, y=data.y_e, z=-data.z_e, mode='markers', marker={'size':2})
    #     corner1_trace = go.Scatter3d(x=corner1.x_e, y=corner1.y_e, z=-corner1.z_e, marker={'size':4})
    #     corner2_trace = go.Scatter3d(x=corner2.x_e, y=corner2.y_e, z=-corner2.z_e, marker={'size':4})
    #     stop1_trace = go.Scatter3d(x=stop1.x_e, y=stop1.y_e, z=stop1.z_e, mode='lines')
    #     stop2_trace = go.Scatter3d(x=stop2.x_e, y=stop2.y_e, z=stop2.z_e, mode='lines')
    #     fig = go.Figure(data=[trace, corner1_trace, corner2_trace, stop1_trace, stop2_trace])
    #     plot(fig, filename='raw_3d.html')

    ''' ---------- Process corners ---------- '''
    print('Processing corners')
    corner1 = corner1[(corner1 < corner1.quantile(0.75)) & (corner1 > corner1.quantile(0.25))]
    corner2 = corner2[(corner2 < corner2.quantile(0.75)) & (corner2 > corner2.quantile(0.25))]

    plt.plot(corner1.x_e, corner1.y_e,'o')
    plt.plot(corner2.x_e, corner2.y_e,'o')
    plt.legend(['corner1', 'corner2'])
    plt.title('corner location after filtration')
    plt.savefig(os.path.join(metafiles_location, 'corner_location.png'), bbox_inches='tight')

    x1 = corner1.x_e.mean()
    y1 = corner1.y_e.mean()
    x2 = corner2.x_e.mean()
    y2 = corner2.y_e.mean()
    xmin = min([x1, x2])
    xmax = max([x1, x2])
    ymin = min([y1, y2])
    ymax = max([y1, y2])
    zref = corner1.z_e.mean()
    zmax = zref + zdelta
    zmin = data.z_e.min() - 1
    print(' {} < x < {} \n {} < y < {} \n z < {}'.format(xmin, xmax, ymin, ymax, zmax))
    
    del corner1, corner2

    # Discard data points for which one marker is out of bound.
    edg_bool = pd.Series(data=[True] * len(data))
    if file_type == 'motive':
        for marker in indivi_markers:
            edg_bool = (edg_bool & edges_bool(indivi_markers[marker].fillna(method='ffill'), xmin, xmax, ymin, ymax, zmin, zmax))
    else:
        # if we only have the pivot point location, discard points where the pivot point is half a span away from boundaries
        # TODO: replace by a calculation of the location of every extreme marker (knowing the body frame vector coords).
        edg_bool = (edg_bool & edges_bool(data.fillna(method='ffill'), xmin, xmax, ymin, ymax, zmin, zmax))

    ''' ---------- Process stop signs ---------- '''
    print('Processing stop signs')
    # Remove Nan
    stop1.fillna(method='ffill', inplace=True)
    stop2.fillna(method='ffill', inplace=True)
    # Interpolate to have values a airplanes times
    if file_type == 'ros':
        aircraft_time = data.time
        stop1_ = data[time+pos].copy()
        stop2_ = data[time+pos].copy()
        for var in pos:
            spl1 = UnivariateSpline(stop1.time + ini_s1 - ini_time, stop1[var])
            stop1_[var] = spl1(data.time)
            spl2 = UnivariateSpline(stop2.time + ini_s2 - ini_time, stop2[var])
            stop2_[var] = spl2(data.time) 
        stop1 = stop1_.copy()
        stop2 = stop2_.copy()
        del stop1_, stop2_

    # find moments when they are in bounds
    stop_sign_bool = 1 - \
             ((stop1.x_e > xmin) & (stop1.x_e < xmax) &
              (stop1.y_e > ymin) & (stop1.y_e < ymax) &
              (stop2.x_e > xmin) & (stop2.x_e < xmax) &
              (stop2.y_e > ymin) & (stop2.y_e < ymax))
    # del stop1_, stop2_

    ''' ---------- Identify beginning and end of each trajectory ---------- '''
    print('Identify beginning and end of each trajectory')
    data = data[edg_bool & stop_sign_bool]
    data.to_csv(input_location.split('.')[0]+'_inbounds_points_only.csv', index=True)
    walls_def = {
        "control_room": (np.array([xmin, 0, 0]), np.array([1, 0, 0])),
        "building_room": (np.array([0, ymin, 0]), np.array([0, 1, 0])),
        "wall": (np.array([xmax, 0, 0]), np.array([-1, 0, 0])),
        "stairs": (np.array([0, ymax, 0]), np.array([0, -1, 0])),
        "ceiling": (np.array([0, 0, zmin]), np.array([0, 0, 1])),
        "floor": (np.array([0, 0, zmax]), np.array([0, 0, -1]))
    }

    # Find where trajectories begin and end 
    traj_beginning_iloc = np.arange(1, len(data))[data.index[1:] - data.index[:-1] > 1]
    traj_beginning_iloc = np.concatenate(([0], traj_beginning_iloc, [len(data) - 1]))
    print(len(traj_beginning_iloc))

    # Build a table to describe trajectories
    keys = ['ini_iloc', 'fin_iloc', 'ini_wall', 'fin_wall', 'ini_wall_dist', 'fin_wall_dist', 'ini_time', 'fin_time']
    trajs = {k: [] for k in keys}
    garbage = {k: [] for k in keys}
    for i in range(len(traj_beginning_iloc) - 1):
        traj = dict()
        traj['ini_iloc'] = traj_beginning_iloc[i]
        traj['fin_iloc'] = traj_beginning_iloc[i + 1] - 1
        if traj['ini_iloc'] == traj['fin_iloc']:
            continue
        ini = data.iloc[traj['ini_iloc']][['x_e', 'y_e', 'z_e']].values
        fin = data.iloc[traj['fin_iloc']][['x_e', 'y_e', 'z_e']].values
        traj['ini_time'] = data.iloc[traj['ini_iloc']]['time']
        traj['fin_time'] = data.iloc[traj['fin_iloc']]['time']
        traj['ini_wall'], traj['ini_wall_dist'] = min_distance(ini, walls_def)
        traj['fin_wall'], traj['fin_wall_dist'] = min_distance(fin, walls_def)
        if traj['ini_wall'] == traj['fin_wall'] or traj['ini_wall'] == 'floor':
            for k in traj:
                garbage[k].append(traj[k])
        else:
            for k in traj:
                trajs[k].append(traj[k])
    trajs = pd.DataFrame.from_dict(trajs)
    garbage = pd.DataFrame.from_dict(garbage)
    trajs.to_csv(os.path.join(metafiles_location, 'traj_info.csv'), index=True)
    garbage.to_csv(os.path.join(metafiles_location, 'discarded_traj_info.csv'), index=True)

    # Get the data in one big dictionary and save csv files
    print('Found {} trajectories'.format(len(trajs['ini_iloc'])))
    print(trajs)
    print('Got rid of {} trajectories'.format(len(garbage['ini_iloc'])))
    print(garbage)
    DATA = {}
    for i in range(len(trajs['ini_iloc'])):
        DATA[i] = data.iloc[trajs['ini_iloc'][i]:trajs['fin_iloc'][i]].copy()
        DATA[i].index = np.arange(len(DATA[i]))
        DATA[i].time = DATA[i].loc[:, 'time'] - DATA[i].loc[0, 'time']
        dest = os.path.join(output_folder, 'traj{}.csv'.format(i))
        DATA[i].to_csv(dest, index=False)

    # Save all markers in trajectory as well
    if file_type == 'motive':
        output_folder_indiv = os.path.join(output_folder,'indiv_markers')
        os.makedirs(output_folder_indiv, exist_ok=True)
        indivi_markers = readMotiveFile(inp_loc, glider_name, all_markers=True)
        for k in indivi_markers:
            indivi_markers[k] = indivi_markers[k][edg_bool & stop_sign_bool]
            for i in range(len(trajs['ini_iloc'])):
                ALL_M = indivi_markers[k].iloc[trajs['ini_iloc'][i]:trajs['fin_iloc'][i]].copy()
                ALL_M.index = np.arange(len(ALL_M))
                ALL_M.time = ALL_M.loc[:, 'time'] - ALL_M.loc[0, 'time']
                dest = os.path.join(output_folder_indiv, '{}_traj{}.csv'.format(k,i))
                ALL_M.to_csv(dest, index=True)

    # Also look at discarded trajectories
    GARB = {}
    if len(garbage) != 0:
        for i in range(len(garbage['ini_iloc'])):
            GARB[i] = data.iloc[garbage['ini_iloc'][i]:garbage['fin_iloc'][i]].copy()
            GARB[i].index = np.arange(len(GARB[i]))
            GARB[i].time = GARB[i].loc[:, 'time'] - GARB[i].loc[0, 'time']

    # Plot all together in an html
    if not noplot:
        trace = []
        for d in DATA:
            name = 'traj'+ str(d) + ':' + trajs['ini_wall'][d] + '->' + trajs['fin_wall'][d]
            trace.append(go.Scatter3d(x=DATA[d].x_e, y=DATA[d].y_e, z=-DATA[d].z_e,
                                      mode='lines+markers', marker={'size': 2}, line={'color': 'blue'},
                                      name=name))
        for g in GARB:
            name = 'garb'+ str(g) + ':' + garbage['ini_wall'][g] + '->' + garbage['fin_wall'][g]
            trace.append(
                go.Scatter3d(x=GARB[g].x_e, y=GARB[g].y_e, z=-GARB[g].z_e, mode='lines+markers',
                             marker={'size': 2}, line={'color': 'red'}, name=name))

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        layout = go.Layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    nticks=8, range=[xmin, xmax]),
                yaxis=dict(
                    nticks=4, range=[ymin, ymax]),
                zaxis=dict(
                    nticks=4, range=[-zmax, -zmin]),
                aspectratio=dict(
                    x=dx/dz,
                    y=dy/dz,
                    z=1
                )))
        fig = go.Figure(data=trace, layout=layout)
        plot(fig, filename=os.path.join(metafiles_location, 'kept_vs_thrown.html'))

    # Plot only selected trajectories
    if not noplot:
        trace = []
        for d in DATA:
            name = 'traj'+ str(d) + ':' + trajs['ini_wall'][d] + '->' + trajs['fin_wall'][d]
            trace.append(go.Scatter3d(x=DATA[d].x_e, y=DATA[d].y_e, z=-DATA[d].z_e,
                                      mode='lines+markers', marker={'size': 2}, name=name))
        fig = go.Figure(data=trace, layout=layout)
        plot(fig, filename=os.path.join(metafiles_location, 'final_dataset.html'))



    trajs = pd.DataFrame.from_dict(trajs)
    garbage = pd.DataFrame.from_dict(garbage)

    # Plot selected trajectories versus time
    if not noplot:
        fld = os.path.join(output_folder, 'traj_plots')
        os.makedirs(fld, exist_ok=True)
        for k in rcParams:
            matplotlib.rc(k, **rcParams[k])
        for d in DATA:
            fig, ax = plt.subplots(2,4, figsize = (20,10))
            name = 'trajectory' + str(d) + ':' + trajs['ini_wall'][d] + '->' + trajs['fin_wall'][d]
            i = 0
            for p in pos:
                ax[0,i].plot(DATA[d].time, DATA[d][p],'+')
                ax[0,i].grid(True)
                ax[0,i].set_ylabel(DEFAULT_TITLE[p])
                ax[0,i].set_xlabel('time in seconds')
                i+=1
            ax[0,3].plot(DATA[d].x_e, -DATA[d].z_e,'+')
            ax[0,3].grid(True)
            ax[0,3].set_xlabel(DEFAULT_TITLE[p])
            ax[0,3].set_ylabel('Altitude -z_e (m)')

            i = 0
            for p in quat:
                ax[1,i].plot(DATA[d].time, DATA[d][p],'+')
                ax[1,i].grid(True)
                ax[1,i].set_ylabel(DEFAULT_TITLE[p])
                ax[1,i].set_xlabel('time in seconds')
                i+=1
            fig.suptitle('Overview of ' + name, fontsize=25, fontweight=10, verticalalignment='bottom')
            fig.tight_layout()        
            fig.savefig(os.path.join(fld, 'traj' + str(d)+'.png'), bbox_inches='tight')
            plt.close(fig)


def aircraft_metadata(aircraft_file):
    mass_cg_estimation(aircraft_file)
    run_inertia_estimation(aircraft_file)

if __name__ == '__main__':
    args = parser.parse_args()
    aircraft_metadata(args['aircraft_file'])
    glider_preprocess(**vars(args))