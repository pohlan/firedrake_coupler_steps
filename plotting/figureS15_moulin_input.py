import os
import sys
import glob
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
from models_main.russel_catchments import get_catchments_russel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cmcrameri.cm as cmc
from firedrake.pyplot import tripcolor
from matplotlib.colors import LogNorm


# plotting parameters
color_map = cmc.managua
plt.rcParams['font.size'] = 29
annotate_fs = 29
lw = 2.5
clims = (20,300)


# timeseries_path = f"test_inversion/forward_beta2_{beta2}e5/time_series.h5"
flowlines_dir = "Greenland_data/profiles/"
outline_GrIS_path = "Greenland_data/gris-outline-imbie-1980_updated_crs.shp"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"
catchments_file = "Greenland_data/Yang_Smith_2016/ds01/catchments_LC80070132013231LGN00_B8.shp"

year = 2020
# xstart,xend = datetime(year-1,4,2), datetime(year,10,30)
xstart, xend = datetime(year,4,2), datetime(year,10,30)

# prepare figure, gridspecs
# fig = plt.figure(figsize=(12,6))
# gs = fig.add_gridspec(6, len(glacier_ids)+1, left=0.11, right=0.92, top=0.95, bottom=0.08, hspace=0.15, wspace=0.1, width_ratios=[15, 15, 15, 15, 15, 1], height_ratios=[1.5, 0.2, 1, 1, 1, 0.8])
fig, ax = plt.subplots(figsize=(15,6))
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.01, top=0.99)

# topax2 = fig.add_subplot(gs[0, 3:6])
# topax2 = fig.add_subplot(gs_top[0,2])

# load model stuff not run-specific
run_index = 713
timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
# B, H, S = load_topography(mesh_)
df_moul, facet_functions, distributed_melt_mask = get_catchments_russel(mesh_)

_, m_raw, _, _, _, _, _, n_idx, _ = load_model_output(timeseries_path)
with h5py.File(timeseries_path, 'r') as h5file:
    Qm_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/Qm/Qm'][()].T
dates_model, i_model = get_model_dates(n_idx, xstart, xend)
V     = df.FunctionSpace(mesh_, "CG", 1)
m     = df.Function(V)
m_distributed   = df.Function(V)
for (i_time, i_m) in enumerate(i_model):
    m_distributed.dat.data[:] = (m_raw[:,i_m]-0.02)/365*2 # remove basal melt 0.02 m/yr
    m_distributed.interpolate(m_distributed*distributed_melt_mask)
    m.dat.data[:] += Qm_raw[:, i_m]/365*2 + m_distributed.dat.data[:]

m.dat.data[m.dat.data_ro<1e-10] = np.nan
im = tripcolor(m, axes=ax, norm=LogNorm(vmin=1e-3, vmax=30), cmap=cmc.batlowK) # cmap=inferno
cbar = fig.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label(r"$m_\mathrm{surface}$" +r" ($\mathrm{m}$)" + "\nintegrated over 2020")

# format
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
ax.set_ylim(-2.585e6,-2.47e6)
ax.set_xlim(-2.4e5,-2e4)

# domain outline
gdf = gpd.read_file(outline_path)
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

# catchments
catchments = gpd.read_file(catchments_file).geometry
catchments = catchments.to_crs(gdf.crs)
catchments.plot(ax=ax, facecolor='none', edgecolor='grey', linewidth=0.3) #, alpha=0.5)
# get moulin coordinates that are exactly on the nodes
coords = hlp.get_coordinates(mesh_, "CG", 1)
meshx = coords[:,0]
meshy = coords[:,1]
x_moulin = np.zeros(len(df_moul.x))
y_moulin = np.zeros(len(df_moul.y))
id_mesh = np.zeros(len(df_moul.y), dtype=int)
i  = 0
for (mx,my) in zip(df_moul.x,df_moul.y):
    ii = np.argmin(np.sqrt((meshx-mx)**2+(meshy-my)**2))
    x_moulin[i] = meshx[ii]
    y_moulin[i] = meshy[ii]
    id_mesh[i] = ii
    i += 1
ax.scatter(x_moulin,y_moulin,2)

# panel label
# ax.annotate("b", [-2.4e5-2e4,-2.47e6+1e4], fontsize=annotate_fs, fontweight="bold", annotation_clip=False)

plt.savefig("plotting/main_figures/fS15.jpg")
