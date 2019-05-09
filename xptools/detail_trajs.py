
# # -*- coding: utf-8 -*-
# from dash import Dash
# from dash_core_components import Graph
# from dash_html_components import Div
# from plotly.graph_objs import Scatter, Figure, layout

import pandas as pd
import csv
import os
import pandas as pd
from vocab import *
import argparse
import matplotlib.pyplot as plt
import pdb

## PARAMETERS
columns = 'pos+quat+eul+vel+rot+frc+mom'

## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('-html', action='store_true',  help="build an html dashboard")
parser.add_argument('-n','--nlines', default=0, help='number of trajectories to plot. Default is all')
parser.add_argument('-c','--columns', default=columns, help='columns to consider, as defined in vocab.py . Strings sperated by + sign')
parser.add_argument('-i','--input_folder', default='camera_ready', help='folder where to find the data')

def create_line(data, traj_name, columns):
    figures = []
    for col in columns:
        trace = Scatter(x=data.time, y=data[col], showlegend=False, mode='markers', line = dict(color=col_colors[col]))
        layout = {'xaxis':{'title':'time (s)'}, 'yaxis':{'title':col_titles[col]}, 'height':200, 'width':300,
        'margin':layout.Margin(l=80, r=30,
        b=40,
        t=30,
        pad=4)}
        fig = Figure(data=[trace], layout=layout)
        figures.append(    
            Div([Graph(id=traj_name + col, figure=fig)], className='forty columns'))
    return(Div(figures, className='container', id=col+'_'+traj_name))


def plot_line(ax_array, data, traj_name, columns, params_dict={}):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    for iax, ax in enumerate(ax_array):
        col = columns[iax]
        data[col] = -data[col] if 'z_e' in col else data[col]
        if col + '_filt' in data.columns:
            data[col+'_filt'] = -data[col+'_filt'] if 'z_e' in col else data[col+'_filt']
            ax.plot(data.time, data[col], '.',color='grey',markersize=2,**params_dict)
            ax.plot(data.time, data[col+ '_filt'], color=col_colors[col],**params_dict)
            ax.legend(['raw signal','low pass filtered'], prop={'size': 5})
        else:
            ax.plot(data.time, data[col], color=col_colors[col],**params_dict)
        ax.set_title(traj_name)
        ax.set_ylabel(col_titles[col])
        ax.set_xlabel('time (s)')
        ax.grid(linewidth=0.25)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return ax_array

if __name__ == '__main__':
    args = parser.parse_args()

    # List columns to consider
    columns = []
    for name in args.columns.split('+'):
        columns += eval(name) 
    print('Showing columns {}'.format(columns))
    ncols = len(columns)

    # List of files to consider
    files= [os.path.join(args.input_folder,f) for f in os.listdir(args.input_folder) if f.endswith('.csv')]
    nlines = int(args.nlines)
    if nlines !=0:
        files = files[:nlines]
    nfiles = len(files)

    # Create web app
    if args.html:
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = Dash(__name__, external_stylesheets=external_stylesheets)

        
        children = []
        for f in files:
            data = pd.read_csv(f)
            children.append(create_line(data, f, columns))

        app.layout = Div(children=children)
        app.run_server(debug=True)

    # Create png
    else:
        fig, axes = plt.subplots(nfiles, ncols, figsize=(ncols*3,nfiles*3))
        for i_f in range(nfiles):
            data = pd.read_csv(files[i_f])
            if nfiles==1:
                plot_line(axes, data, files[i_f].split('/')[-1], columns) 
            else:
                plot_line(axes[i_f], data, files[i_f].split('/')[-1], columns) 
        plt.tight_layout()
        plt.subplots_adjust(hspace=.5)
        # plt.show()
        try:
            plt.savefig('data_summary.png',dpi=220)
        except:
            plt.savefig('data_summary.png',dpi=180)