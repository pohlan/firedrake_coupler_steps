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
from datetime import datetime, timedelta
import cmcrameri.cm as cmc
from firedrake.pyplot import tripcolor
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter

# ToDo: plot mu instead of beta2...

flowlines_dir = "Greenland_data/profiles/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

# plotting parameters
color_map = cmc.managua
plt.rcParams['font.size'] = 22
annotate_fs = 21
vmi, vma = 1.5e-5, 1.05e-4


fig = plt.figure(figsize=(19,5))
gs_top = fig.add_gridspec(1, 3, hspace=0.15, wspace=0.04, width_ratios=[1,0.05,1], left=0.0,right=0.99,top=0.88,bottom=0.03)

ax1 = fig.add_subplot(gs_top[0,0])
ax2 = fig.add_subplot(gs_top[0,2])


# baseline
chk_file = f"test_inversion/beta2_opt_1e-2.0_run304.h5"
with CheckpointFile(chk_file, 'r') as afile:
        mesh_ = afile.load_mesh()
        beta2 = afile.load_function(mesh_, name="beta2")

# Uhat and Nhat
B, H, S = load_topography(mesh_)
area = df.assemble(df.Constant(1)*df.dx(domain=mesh_))
H_mean = df.assemble(H*df.dx)/area
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

beta2.dat.data[:] = beta2.dat.data_ro * (1/Uhat) * (1/Nhat)**(1.2)
im = tripcolor(beta2, axes=ax1, cmap=cmc.davos, vmin=vmi, vmax=vma)
# norm=colors.LogNorm(vmin=1e-5, vmax=0.1)
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label(r"$\mu$ $\mathrm{a\,m^{-1}\,Pa^{-1/5}}$")
# use scientific notation x1e-4
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_major_formatter(formatter)
remove_axes(ax1)
gdf = gpd.read_file(outline_path)
gdf.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=1, label='Domain')
ax1.annotate("Baseline", xy=[0.5,1.1], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")

# baseline
chk_file = f"test_inversion/beta2_opt_1e-2.0_run477.h5"
with CheckpointFile(chk_file, 'r') as afile:
        mesh_ = afile.load_mesh()
        beta2 = afile.load_function(mesh_, name="beta2")

beta2.dat.data[:] = beta2.dat.data_ro * (1/Uhat) * (1/Nhat)**(1.2)
im = tripcolor(beta2, axes=ax2, cmap=cmc.davos, vmin=vmi, vmax=vma)
cbar = fig.colorbar(im, ax=ax2)
cbar.set_label(r"$\mu$ $\mathrm{a\,m^{-1}\,Pa^{-1/5}}$")
vmin, vmax = cbar.mappable.get_clim()
# use scientific notation x1e-4
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_major_formatter(formatter)
remove_axes(ax2)
gdf = gpd.read_file(outline_path)
gdf.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=1, label='Domain')
ax2.annotate("Reduced sheet flow", xy=[0.5,1.1], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")


panel_letter_annotation(ax1, 0, annotate_fs)
panel_letter_annotation(ax2, 1, annotate_fs)

plt.savefig(f"plotting/main_figures/fS01c.jpg")

