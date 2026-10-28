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
plt.rcParams['font.size'] = 41
annotate_fs = 41
lw = 2.5
clims = (20,300)

# timeseries_path = f"test_inversion/forward_beta2_{beta2}e5/time_series.h5"
flowlines_dir = "Greenland_data/profiles/"
outline_GrIS_path = "Greenland_data/gris-outline-imbie-1980_updated_crs.shp"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"
catchments_file = "Greenland_data/Yang_Smith_2016/ds01/catchments_LC80070132013231LGN00_B8.shp"

year = 2020
xstart, xend = datetime(year-1,4,2), datetime(year-1,10,30)

# prepare figure, gridspecs
fig, ax = plt.subplots(figsize=(15,6))
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.01, top=0.99)

# load model stuff not run-specific
run_index = 715
timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
V     = df.FunctionSpace(mesh_, "CG", 1)

fig = plt.figure(figsize=(28, 8))
gs = fig.add_gridspec(1, 3, left=0, right=0.94, top=0.82, bottom=0.0, hspace=-0.1, wspace=0, width_ratios=[1,1,0.12])


with h5py.File(timeseries_path, 'r') as h5file:
    Mfr_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/M_fr/M_fr'][()].T

# load model output
us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, _, n_idx, _ = load_model_output(timeseries_path)
for (i_s,season) in enumerate(["July 2020", "January 2021"]):
    idx = {'January 2021': int(7.05*365/2), 'July 2020': int(6.53*365/2)}[season]
    ax = fig.add_subplot(gs[0,i_s])

    M_fr  = df.Function(V)
    M_fr.dat.data[:] = Mfr_raw[:,idx]

    cl = tripcolor(M_fr, axes=ax, cmap=cmc.lajolla, norm=LogNorm(vmin=5e-3, vmax=0.1))
    if (i_s == 0):
        cax1 = fig.add_subplot(gs[0, 2])
        cbar = fig.colorbar(cl, cax=cax1, label=r"$m_\mathrm{friction}\,(\mathrm{m\,a^{-1}})$")

        # manually define position and size of colorbar
        pos = cax1.get_position()
        cax1.set_position([
            pos.x0 + 0.2 * pos.width,
            pos.y0 + 0.25 * pos.height,
            0.2*pos.width,
            0.5*pos.height
        ])

    # domain outline
    gdf = gpd.read_file(outline_path)
    gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Domain')
    ax.annotate(season, xy=[0.5,1.15], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    ax.annotate(f"{idx_to_letter(i_s)}", xy=(0.03,1.05), xycoords="axes fraction", fontsize=annotate_fs*0.9, fontweight="bold")
    remove_axes(ax)

plt.savefig(f"plotting/main_figures/fS17.jpg")
