import os
import firedrake as df
from firedrake.__future__ import interpolate
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor, triplot
import xarray as xr
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from firedrake.checkpointing import CheckpointFile
from datetime import datetime, timedelta
import re

n_idx = 1050

# load model files
with CheckpointFile("step_10b/results/time_series.h5", 'r') as afile:
    mesh = afile.load_mesh()
    V = df.FunctionSpace(mesh, "CG", 1)
    v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
    X = df.assemble(interpolate(mesh.coordinates,v_dg))
    meshx = X.dat.data_ro[:,0]
    meshy = X.dat.data_ro[:,1]
    phi_model = np.zeros((len(meshx), n_idx))
    Us_model = np.zeros((len(meshx), n_idx))
    for i in range(n_idx):
        phi_model[:,i] = afile.load_function(mesh, "phi", idx=i).vector()[:]
        Us_model[:,i]  = afile.load_function(mesh, "Us", idx=i).vector()[:]

# generate time vector
start_date = datetime(2016, 1, 1)
dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]



# load observations and interpolate onto mesh
# V = df.FunctionSpace(mesh, "CG", 1)
# v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
# X = df.assemble(interpolate(mesh.coordinates,v_dg))
# meshx = X.dat.data_ro[:,0]
# meshy = X.dat.data_ro[:,1]
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
files   = os.listdir(vel_dir)

# extract datetime object from filenames and sort after date
date_pattern = r'\d{2}[A-Z][a-z]{2}\d{2}'
format_string = "%d%b%y"
date_list = []
for f in files:
    date_str    = re.search(date_pattern, f).group(0)
    date_object = datetime.strptime(date_str, format_string)
    date_list.append(date_object)
pairs = zip(date_list, files)
sorted_dates, sorted_files = zip(*sorted(pairs))

# read files in order
n_months = len(files)
Us_obs = np.zeros((len(meshx),n_months))
for (i,f) in enumerate(sorted_files[:n_months]):
    r = gu.Raster(f"{vel_dir}{f}")
    delta = r.res[0]*2
    r.crop([min(meshx)-delta, min(meshy)-delta, max(meshx)+delta, max(meshy)+delta], inplace=True)
    Us_obs[:,i] = r.interp_points((meshx,meshy))

# def plot_vel(k):
#     B = df.Function(V)
#     B.vector()[:] = Us_obs[:,k]
#     fig, axes = plt.subplots()
#     cl = tripcolor(B, axes=axes)
#     axes.set_title(f"{sorted_dates[k]}")
#     fig.colorbar(cl)
#     plt.savefig("B.jpg")
# k = 56
# plot_vel(k)


# xi, yi = -206185, -2.49758e6
# xi, yi = -218312.472667319,-2514023.04736881
xi, yi = -215217.697420904,-2510443.87251861
# xi, yi = -203042.0,-2507072.0
# xi, yi = -192320.0,-2508747.0
p = np.argmin(np.sqrt((meshx-xi)**2+(meshy-yi)**2))
plt.figure()
plt.plot(dates_model[:-1], Us_model[p,1:], label="model")
plt.plot(sorted_dates[:n_months], Us_obs[p,:], label="obs")
plt.legend()
plt.savefig("B.jpg")






