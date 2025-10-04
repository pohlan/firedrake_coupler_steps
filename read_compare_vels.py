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

run_index = 41

# load model files
with CheckpointFile(f"parameter_runs/run_{run_index}/time_series.h5", 'r') as afile:
    mesh = afile.load_mesh()
    V = df.FunctionSpace(mesh, "CG", 1)
    v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
    X = df.assemble(interpolate(mesh.coordinates,v_dg))
    meshx = X.dat.data_ro[:,0]
    meshy = X.dat.data_ro[:,1]
    n_idx = len(afile.get_timestepping_history(mesh, "phi")['index'])
    phi_model = np.zeros((len(meshx), n_idx))
    Us_model = np.zeros((len(meshx), n_idx))
    for i in range(n_idx):
        phi_model[:,i] = afile.load_function(mesh, "phi", idx=i).vector()[:]
        Us_model[:,i]  = afile.load_function(mesh, "Us", idx=i).vector()[:]

# generate time vector
start_date = datetime(2015, 11, 1)
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

# original coords, mix of different glaciers
# coords = [[-223885, -2.49326e6],
#           [-205468, -2.49596e6],
#           [-225130, -2.50269e6],
#           [-216304, -2.50337e6],
#           [-218312.472667319,-2514023.04736881],
#           [-215217.697420904,-2510443.87251861],
#           [-203042.0,-2507072.0],
#           [-192320.0,-2508747.0]]
# gl = ""

# glacier #1 (furthest north)
coords = [[-223885, -2.49326e6],
          [-217000, -2.49416e6],
          [-211000, -2.49506e6],
          [-205468, -2.49596e6],
          [-199000, -2.49676e6],
          [-193000, -2.49760e6]]
gl = 1

# glacier #2
# coords  = [[-225130, -2.50269e6],
#           [-215304, -2.50337e6],
#           [-203042.0,-2507072.0],
#           [-192320.0,-2508747.0]]
# gl = 2

# glacier #3
# coords  = [[-215017, -2.52269e6],
#            [-210017, -2.52700e6],
#            [-203042.0,-2529072.0],
#            [-199320.0,-2525747.0],
#            [-192320.0,-2520747.0],
#            [-192320.0,-2530747.0]]
# gl = 3

plt.figure(figsize=(12,14))
for (i,(xi,yi)) in enumerate(coords):
    p = np.argmin(np.sqrt((meshx-xi)**2+(meshy-yi)**2))
    ax = plt.subplot(4,2,i+1)
    plt.plot(dates_model[:-1], Us_model[p,1:], label="model")
    plt.plot(sorted_dates[:n_months], Us_obs[p,:], label="obs")
    plt.title(f"{i}", pad=2.0)
    plt.ylabel("Surface speed (m/yr)")
    # plt.ylim(15,225)
    plt.legend()
plt.savefig(f"parameter_runs/plots/run_{run_index}_gl{gl}.jpg")

# plot map
x0, xend = -2.35e5,-1.75e5
y0, yend = -2.55e6, -2.488e6
backg = gu.Raster(f"{vel_dir}{files[0]}")
delta = backg.res[0]*2
backg.crop([x0, y0, xend, yend], inplace=True)
plt.figure()
backg.plot()
for(i,(x_crd,y_crd)) in enumerate(coords):
    plt.scatter([x_crd],[y_crd], color="black")
    plt.annotate(text=f"{i}", xy=(x_crd+5e2,y_crd+5e2), fontsize=14)
plt.savefig(f"parameter_runs/plots/map_gl{gl}.jpg")


# # plot ice thickness
# gr_thick = gu.Raster("NETCDF:Greenland_data/BedMachineGreenland-v5.nc:thickness")
# delta = gr_thick.res[0]*2
# gr_thick.crop([x0, y0, xend, yend], inplace=True)
# plt.figure()
# gr_thick.plot(cbar_title="Ice thickness (m)")
# plt.clabel("Ice thickness (m)")
# plt.savefig("parameter_runs/plots/thickness_map.jpg")
