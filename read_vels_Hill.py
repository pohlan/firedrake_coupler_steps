import firedrake as df
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from firedrake.checkpointing import CheckpointFile
from datetime import datetime, timedelta

dir_SHAK = "step_11b/"
dir_hill = "step_10c_trans_32/"

# load model files
# with CheckpointFile(f"{dir}time_series_hill_geom.h5", 'r') as afile:
# with CheckpointFile(f"{dir}time_series_no_channels.h5", 'r') as afile:
def get_phi(file):
    with CheckpointFile(file, 'r') as afile:
        mesh = afile.load_mesh()
        V = df.FunctionSpace(mesh, "CG", 1)
        v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
        X = df.assemble(df.interpolate(mesh.coordinates,v_dg))
        meshx = X.dat.data_ro[:,0]
        meshy = X.dat.data_ro[:,1]
        n_idx = len(afile.get_timestepping_history(mesh, "pw_pi")['index'])
        phi_model = np.zeros((len(meshx), n_idx))
        for i in range(n_idx):
            phi_model[:,i] = afile.load_function(mesh, "pw_pi", idx=i).dat.data_ro
        return meshx, meshy, n_idx, phi_model

meshx, meshy, n_hill, phi_hill = get_phi(f"{dir_hill}time_series.h5")
meshx, meshy, n_hill_no_ch, phi_hill_no_ch = get_phi(f"{dir_hill}time_series_no_channels.h5")
meshx, meshy, n_SHAKTI, phi_SHAKTI = get_phi(f"{dir_SHAK}time_series_hill_geom.h5")
meshx, meshy, n_SHAKTI_hill_q, phi_SHAKTI_hill_q = get_phi(f"{dir_SHAK}time_series_hill_with_M.h5")

# generate time vector
# start_date = datetime(2016, 1, 1)
# dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]


# original coords, mix of different glaciers
coords = [[15e3, 13e3],
          [30e3, 13e3],
          [70e3, 13e3]]

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 33})
plt.figure(figsize=(25,14))
for (i,(xi,yi)) in enumerate([coords[0]]):
    p = np.argmin(np.sqrt((meshx-xi)**2+(meshy-yi)**2))
    ax = plt.subplot(1,1,1)
    for (n_idx,phi,lab) in zip([n_hill,n_hill_no_ch,n_SHAKTI,n_SHAKTI_hill_q],[phi_hill,phi_hill_no_ch,phi_SHAKTI,phi_SHAKTI_hill_q],["GlaDS +","GlaDS + without channels","SHAKTI","SHAKTI with k_s"]):
        dates_model = np.linspace(0,365*2,n_idx)
        plt.plot(dates_model[:-1], phi[p,1:], label=lab, lw=4)
    plt.title("Water pressure time series 15 km away from the terminus")
    plt.ylabel("Pw/Pi (-)")
    plt.xlabel("Day of the year")
    plt.xlim(0,365)
    plt.ylim(0,1.1)
    plt.legend()
    plt.margins(0)
plt.savefig(f"{dir_hill}time_series_all.jpg",dpi=600)
