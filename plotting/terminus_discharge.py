import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from datetime import datetime, timedelta
import firedrake as df
# from firedrake.pyplot import tripcolor
import cmcrameri.cm as cmc
# import matplotlib.colors as colors
from shapely.geometry import box
import fiona

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
xc = [[xcoords[0]]]
yc = [[ycoords[0]]]
x0 = 0.5*xcoords[0] + 0.5*xcoords[1]

data_dir = "/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/Greenland_data/russel/"
hydro_file = f"{data_dir}hydrological_outlets_russel.gpkg"
hydro_points = fiona.open(hydro_file, mode="r")
hydro_coords = np.array([i['geometry']['coordinates'] for i in list(hydro_points.values())])
hydro_points.close()

print(hydro_coords)

run_indices_suit = list(range(601,725))
m_b_0 = 0.02
run_indices = [314,478,483]
model_labels  = ["Baseline", "Reduced sheet flow", "483", r"$h_r=8$ m", "any", "any"]
cols = ["coral","cornflowerblue","yellowgreen","palevioletred", "red"]
glacier_names = ["Isunnguata Sermia", "Russel", "Ørkendalen", "Isorlersuup", "Glacier 5"]

xstart, xend = datetime(2020,4,1),datetime(2021,6,1)
# xstart,xend = datetime(2018,5,1),datetime(2022,6,1)
xstart_winter, xend_winter = datetime(2020,10,1), datetime(2021,4,30)
xstart_summer, xend_summer = datetime(2020,5,1), datetime(2020,9,30)

outline_path   = "Greenland_data/russel/russel_domain.gpkg"
gdf = gpd.read_file(outline_path)
vel_dir = "Greenland_data/velocity/monthly/"

# plotting parameters
plt.rcParams['font.size'] = 16
annotate_fs = 18
lw = 2
markersz = 30
# colors = plt.cm.viridis(np.linspace(0.0, 1.0, 5))
colors = cmc.berlin(np.linspace(0.0, 1.0, 5))

# load model input independent of run
timeseries_path = f"parameter_runs/run_477/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
coords = smesh_.coordinates.dat.data  # mesh coordinates
B, H, S = load_topography(mesh_, sig=5)

fig = plt.figure(figsize=(20, 5))
gs  = fig.add_gridspec(1, 3, left=0.07, right=0.95, hspace=0.05, top=0.9, bottom=0.15, width_ratios=[1,1,1])

# gs_top = gs[0].subgridspec(1, 5, hspace=0.0, wspace=0.04, )
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])


idx_outflow = get_outflow_index_skeleton_mesh(mesh_, smesh_)
idx_throughflow = get_interior_flux_edges(mesh_, smesh_, x0)

x, y = df.SpatialCoordinate(smesh_)

V_sub = df.FunctionSpace(smesh_, "DG", 0)
Fx   = df.Function(V_sub).interpolate(x)
Fy   = df.Function(V_sub).interpolate(y)
Fout = df.Function(V_sub)
termini_outlow_idx = []
for (i_gl, (xi,yi)) in enumerate(hydro_coords):
    dist_to_terminus = np.sqrt((Fx.dat.data_ro[idx_outflow]-xi)**2 + (Fy.dat.data_ro[idx_outflow]-yi)**2)
    # idx_terminus = idx_outflow[np.argpartition(dist_to_terminus, 7)[:7]]
    idx_terminus = idx_outflow[np.where(dist_to_terminus < 5e3)[0]]
    if i_gl == 3:
        idx_terminus = idx_terminus[np.where(Fy.dat.data_ro[idx_terminus] > -2518780)[0]]
    if i_gl == 4:
        idx_terminus = idx_terminus[np.where(Fy.dat.data_ro[idx_terminus] < -2518780)[0]]
    termini_outlow_idx.append(idx_terminus)
    Fout.dat.data[idx_terminus] = i_gl+1
    print(idx_terminus)
df.VTKFile("Fout.pvd").write(Fout)

for (i_r,(run_index,model_label,col,ax)) in enumerate(zip(run_indices,model_labels,cols,[ax1,ax2,ax3])):
    print(run_index)
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, S_raw, n_idx, _ = load_model_output(timeseries_path)
    dates_model, i_model = get_model_dates(n_idx, xstart, xend)

    Q_termini = np.zeros((len(i_model),len(hydro_coords)))


    for (i_time, i_m) in enumerate(i_model):
        # output channels
        for (i_gl, idx) in enumerate(termini_outlow_idx):
            Q_termini[i_time,i_gl] = np.sum(Q_raw[idx, i_m])

    ax.set_prop_cycle(cycler(color=colors))
    for (i_gl,(i,col)) in enumerate(zip([0,4,1,2,3],cols)):
        # gl = [0,4,1,2,3][i]
        ax.plot(dates_model,Q_termini[:,i]/365*1e-9, label=glacier_names[i_gl], lw=lw)
    ax.set_title(model_label)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.set_xlabel("Month of 2020/2021")
    ax.set_ylim(-0.001,0.054)
    panel_letter_annotation(ax, i_r, annotate_fs, xyc=(0.03, 1.04))

    if i_r == 0:
        ax.legend()
        ax.set_ylabel(r"Terminus discharge ($\mathrm{km^3\,d^{-1}}$)")


plt.savefig(f"plotting/main_figures/f06.jpg")
