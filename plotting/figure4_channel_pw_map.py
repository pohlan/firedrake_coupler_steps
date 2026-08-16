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
from firedrake.pyplot import tripcolor
import cmcrameri.cm as cmc
import matplotlib.colors as colors

run_indices = [327,477]
model_labels  = ["Baseline", "Reduced \nsheet flow"]
Q_min     = 10
# vmax = 5e10

flowlines_path = "Greenland_data/russel/flowlines.gpkg"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"
vel_dir = "Greenland_data/velocity/monthly/"

# plotting parameters
plt.rcParams['font.size'] = 28
annotate_fs = 28

# load model input independent of run
timeseries_path = f"parameter_runs/run_{run_indices[0]}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
coords = smesh_.coordinates.dat.data  # mesh coordinates
B, H, S = load_topography(mesh_, sig=5)


def get_Q_lines(Q_vals, ax, cax):
    # Get cell-vertex connectivity using CG function space
    V_CG = df.FunctionSpace(smesh_, "CG", 1)
    cells = V_CG.cell_node_list
    # Create line segments for mesh edges, each colored by Q value
    lines = []
    colors = []
    # Determine if cells are edges (2 vertices) or triangles (3 vertices)
    num_vertices_per_cell = cells.shape[1]
    for cell_idx, cell in enumerate(cells):
        q_val = Q_vals[cell_idx]
        vertices = cell
        if num_vertices_per_cell == 2:
            # Edges: just draw the single line segment
            p1 = coords[vertices[0]]
            p2 = coords[vertices[1]]
            lines.append([p1[:2], p2[:2]])
            colors.append(q_val)
        elif num_vertices_per_cell == 3:
            # Triangles: draw all 3 edges
            edges = [(vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])]
            for edge in edges:
                p1 = coords[edge[0]]
                p2 = coords[edge[1]]
                lines.append([p1[:2], p2[:2]])
                colors.append(q_val)
    # Create LineCollection and add to plot
    # lc = mc.LineCollection(lines, cmap='Greys',
                        #    norm=plt.Normalize(vmin=np.nanmin(Q_vals)*(1), vmax=Q_min*1000))
    lc = mc.LineCollection(lines, cmap=cmc.lajolla_r, norm=plt.Normalize(vmin=Q_min, vmax=140), joinstyle='round', capstyle='round')
                        #    norm=plt.Normalize(vmin=np.nanmin(Q_vals)*(1), vmax=Q_min))
    lc.set_array(np.array(colors))
    lc.set_linewidth(3)
    cbar = fig.colorbar(lc, ax=cax)
    cbar.set_label(r"$Q\,(\mathrm{m^3 s^{-1}})$")

    return lc

    # remove spines (box around plot)
    for spine in ax.spines.values():
        spine.set_visible(False)

# Create figure and plot
fig = plt.figure(figsize=(30, 14))
gs = fig.add_gridspec(2, 3, left=0.12, right=0.93, top=0.92, bottom=0, hspace=0.0, wspace=0.1, width_ratios=[1,1,0.12], height_ratios=[1,1])


for (i_r,run_index) in enumerate(run_indices):
    # load model output
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, n_idx, _ = load_model_output(timeseries_path)
    for (i_s,season) in enumerate(["January 2021","July 2021"]):
        idx = {'January 2021': int(7.05*365/2), 'July 2021': int(7.53*365/2)}[season]
        ax = fig.add_subplot(gs[i_r,i_s])

        # extract q for the specified timestep
        V   = df.FunctionSpace(mesh_, "CG", 1)
        q_vec_x = q_raw[::2, idx]
        q_vec_y = q_raw[1::2, idx]
        q = df.Function(V)
        q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2) / (3600*24*365)



        # h = df.Function(V)
        # h.dat.data[:] = h_raw[:,idx]

        cl = tripcolor(q, axes=ax, cmap=cmc.lapaz, norm=colors.LogNorm(vmin=1e-5, vmax=0.1))
        cax1 = fig.add_subplot(gs[0, 2])
        cbar = fig.colorbar(cl, ax=cax1, label=r"$q\,(\mathrm{m^2 s^{-1}})$")

        # domain outline
        gdf = gpd.read_file(outline_path)
        gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Domain')

        # extract Q for the specified timestep
        Q_vals = Q_raw[:, idx] / (3600*24*365)
        Q_vals[np.where(Q_vals<Q_min)] = np.nan

        cax2 = fig.add_subplot(gs[1, 2])
        lc = get_Q_lines(Q_vals, ax, cax2)
        ax.add_collection(lc)

        remove_axes(ax)
        remove_axes(cax1)
        remove_axes(cax2)

        if i_s == 0:
            model_label = model_labels[i_r]
            ax.annotate(model_label, xy=[-0.15,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        if i_r == 0:
            ax.annotate(season, xy=[0.5,1.1], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        ax.annotate(f"{idx_to_letter(i_r*2+i_s)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.9, fontweight="bold")



# # add markers and annotations
# colors = ["coral","cornflowerblue","yellowgreen","palevioletred"]
# annotate_offsets = [[-3e3,6e3],[-1.2e4,-2.5e3],[-1.1e4,-1e4],[-1.1e4,-1e4]]
# gdf_flowlines = gpd.read_file(flowlines_path)
# gls = [1,3,4,5]
# d_upglacier = [10e3,10e3,10e3,10e3]
# marker_x = []
# marker_y = []
# for (gl,d,col,strng,offset_crd) in zip(gls,d_upglacier,colors,["(1)","(2)","(3)","(4)"],annotate_offsets):
#     fl = [0,4,1,2,3][gl-1]
#     coords = list(gdf_flowlines.geometry[fl].coords)
#     dists = segment_lengths(gdf_flowlines.geometry[fl])
#     marker_x.append(coords[np.argmin(abs(dists-d))][0])
#     marker_y.append(coords[np.argmin(abs(dists-d))][1])
#     ax.annotate(strng,[marker_x[-1],marker_y[-1]], xytext=[marker_x[-1]+offset_crd[0],marker_y[-1]+offset_crd[1]], c=col, fontsize=29)

# plt.scatter(marker_x, marker_y, 210, c=colors, edgecolors="black", linewidths=1)

plt.tight_layout()
plt.savefig("plotting/main_figures/f03.jpg")
plt.savefig(f"plotting/output_heatmap/Q_map_{run_indices[0]}_{run_indices[1]}_q.png", dpi=150)
