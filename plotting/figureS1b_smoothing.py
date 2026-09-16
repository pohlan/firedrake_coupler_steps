import os
import sys
import glob
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from firedrake.pyplot import tripcolor

# timeseries_path = f"test_inversion/forward_beta2_{beta2}e5/time_series.h5"
flowlines_dir = "Greenland_data/profiles/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

# plotting parameters
color_map = cmc.managua
plt.rcParams['font.size'] = 22
annotate_fs = 23

# load model stuff not run-specific
timeseries_path = f"parameter_runs/run_478/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
B, H, S = load_topography(mesh_)

# prepare figure, gridspecs
fig = plt.figure(figsize=(20,10))
# gs = fig.add_gridspec(6, 1, left=0.11, right=0.92, top=0.95, bottom=0.08, hspace=0.15, wspace=0.1)
gs_top = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.04, width_ratios=[1,0.05,1], left=0.0,right=0.97,top=0.88,bottom=0.03)

# plot bed topography


def get_topo(file):
    r = gu.Raster(file)
    # delta = r.res[0]*2
    r.crop([-2.4e5, -2.585e6, 0, -2.47e6], inplace=True)
    outline = gu.Vector(outline_path)
    mask   = ~outline.create_mask(r)
    r.set_mask(mask)
    return r

##################
# bed topography #
##################

ax1 = fig.add_subplot(gs_top[0,0])
ax2 = fig.add_subplot(gs_top[0,2])

# smooth
im = tripcolor(B, axes=ax1, cmap="terrain")
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label("Bed elevation (m)")
vmin, vmax = cbar.mappable.get_clim()

remove_axes(ax1)
gdf = gpd.read_file(outline_path)
gdf.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

# original
r_bed = get_topo("NETCDF:Greenland_data/BedMachineGreenland-v5.nc:bed")
data    = r_bed.data.squeeze()
x, y    = r_bed.coords(grid=False)
im = ax2.pcolormesh(x, np.flip(y), data, cmap="terrain", vmin=vmin, vmax=vmax)
cbar = fig.colorbar(im, ax=ax2)
cbar.set_label("Bed elevation (m)")
remove_axes(ax2)
gdf = gpd.read_file(outline_path)
gdf.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=1, label='Domain')


ax1.annotate("Smooth", xy=[0.5,1.2], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
ax2.annotate("Original", xy=[0.5,1.2], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")

panel_letter_annotation(ax1, 0, annotate_fs)
panel_letter_annotation(ax2, 1, annotate_fs)

##################
# Ice thickness  #
##################

ax1 = fig.add_subplot(gs_top[1,0])
ax2 = fig.add_subplot(gs_top[1,2])

# smooth
im = tripcolor(H, axes=ax1, cmap=cmc.oslo)
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label("Ice thickness (m)")
vmin, vmax = cbar.mappable.get_clim()

remove_axes(ax1)
gdf = gpd.read_file(outline_path)
gdf.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

# original
r_bed = get_topo("NETCDF:Greenland_data/BedMachineGreenland-v5.nc:thickness")
data    = r_bed.data.squeeze()
x, y    = r_bed.coords(grid=False)
im = ax2.pcolormesh(x, np.flip(y), data, cmap=cmc.oslo, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(im, ax=ax2)
cbar.set_label("Ice thickness (m)")
remove_axes(ax2)
gdf = gpd.read_file(outline_path)
gdf.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

panel_letter_annotation(ax1, 2, annotate_fs)
panel_letter_annotation(ax2, 3, annotate_fs)


plt.savefig(f"plotting/main_figures/fS01b.jpg")
