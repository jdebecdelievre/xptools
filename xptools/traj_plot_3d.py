from xptools.motive_data_helpers import *
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import csv
import os
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from xptools.inertia_estimation import run_inertia_estimation, mass_cg_estimation

def traj_plot_3d(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    trace = []
    for f in files:
        data = pd.read_csv(f)
        name = f
        trace.append(go.Scatter3d(x=data.x_e, y=data.y_e, z=-data.z_e,
                mode='lines+markers', marker={'size': 2}, name=name))
    fig = go.Figure(data=trace)
    plot(fig, filename=os.path.join(os.path.join(folder, 'dataset.html')))

if __name__=='__main__':
    traj_plot_3d('.')
