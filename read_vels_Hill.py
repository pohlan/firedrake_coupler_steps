import firedrake as df
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from firedrake.checkpointing import CheckpointFile
from datetime import datetime, timedelta

# load model files
with CheckpointFile(f"step_10c_trans_32/time_series.h5", 'r') as afile:
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

# generate time vector
# start_date = datetime(2016, 1, 1)
# dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]

dates_model = np.arange(0,2*n_idx,2)

# original coords, mix of different glaciers
coords = [[15e3, 13e3],
          [30e3, 13e3],
          [70e3, 13e3]]

plt.figure(figsize=(12,14))
for (i,(xi,yi)) in enumerate(coords):
    p = np.argmin(np.sqrt((meshx-xi)**2+(meshy-yi)**2))
    ax = plt.subplot(4,2,i+1)
    plt.plot(dates_model[:-1], phi_model[p,1:], label="model")
    # print(phi_model[p,1:])
    plt.title(f"{i}", pad=2.0)
    plt.ylabel("Surface speed (m/yr)")
    # plt.xlim(365,2*365)
    plt.ylim(0,1.5)
    plt.legend()
plt.savefig(f"step_10c_trans_32/time_series.jpg")
