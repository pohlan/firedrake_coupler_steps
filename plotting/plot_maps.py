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

run_index = 191
Q_min     = 9.46e8
# idx = int(2.53*365/2)  # timestep idx to plot (here max discharge)
# idx = int(3*365/2)  # timestep idx to plot (here winter)
# idx = int(4.53*365/2)  # timestep idx to plot (here max discharge)
season = ['winter','summer'][0]
idx = {'winter': int(5*365/2), 'summer': int(4.53*365/2)}[season]
vmax = 6e9

timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
flowlines_path = "Greenland_data/russel/flowlines.gpkg"
outline_path   = "Greenland_data/russel/russel_domain_less_points.gpkg"
vel_dir = "Greenland_data/velocity/monthly/"

mesh_, smesh_ = get_meshes(timeseries_path)

us_raw, m_raw, phi_raw, q_raw, Q_raw, n_idx = load_model_output(timeseries_path)

# Plot Q_raw as a colored mesh for a specific timestep
B, H, S = load_topography(mesh_, sig=1)

# Extract Pw/Pi for the specified timestep
V   = df.FunctionSpace(mesh_, "CG", 1)
phi = df.Function(V)
phi.dat.data[:] = phi_raw[:, idx]
pw_pi = df.Function(V)
pw_pi.interpolate((phi-1000*9.81*B)/(910*9.81*H))
# pw_pi_vals = pw_pi.dat.data_ro

# Extract Q for the specified timestep
Q_vals = Q_raw[:, idx]
Q_vals[np.where(Q_vals<Q_min)] = np.nan

# Get mesh coordinates
coords = smesh_.coordinates.dat.data

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
lc = mc.LineCollection(lines, cmap='Greys',
                       norm=plt.Normalize(vmin=np.nanmin(Q_vals)*(1), vmax=vmax*1000))
lc.set_array(np.array(colors))
lc.set_linewidth(3)


plt.rcParams['font.size'] = 23

# Create figure and plot
fig, ax = plt.subplots(figsize=(10, 9))

# pw_pi
# cmap: lapaz or oleron..
cl = tripcolor(pw_pi, axes=ax, cmap=cmc.lapaz, vmin=-0.5, vmax=1.0)
fig.colorbar(cl, label="Pw / Pi")

# Q
ax.add_collection(lc)
# cbar = fig.colorbar(lc, ax=ax, label='Q (discharge)')

# domain outline
gdf = gpd.read_file(outline_path)
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Domain')

# ax.autoscale()
ax.set_aspect('equal')


ax.set_xlim(-2.4e5,-1.6e5)
ax.set_ylim(-2.565e6,-2.47e6)
ax.set_xticks([])
ax.set_yticks([])
# Remove spines (box around plot)
for spine in ax.spines.values():
    spine.set_visible(False)


# add markers and annotations
colors = ["coral","cornflowerblue","yellowgreen","palevioletred"]
annotate_offsets = [[-3e3,6e3],[-1.2e4,-2.5e3],[-1.1e4,-1e4],[-1.1e4,-1e4]]
gdf_flowlines = gpd.read_file(flowlines_path)
gls = [1,3,4,5]
d_upglacier = [10e3,10e3,10e3,10e3]
marker_x = []
marker_y = []
for (gl,d,col,strng,offset_crd) in zip(gls,d_upglacier,colors,["(1)","(2)","(3)","(4)"],annotate_offsets):
    fl = [0,4,1,2,3][gl-1]
    coords = list(gdf_flowlines.geometry[fl].coords)
    dists = segment_lengths(gdf_flowlines.geometry[fl])
    marker_x.append(coords[np.argmin(abs(dists-d))][0])
    marker_y.append(coords[np.argmin(abs(dists-d))][1])
    ax.annotate(strng,[marker_x[-1],marker_y[-1]], xytext=[marker_x[-1]+offset_crd[0],marker_y[-1]+offset_crd[1]], c=col, fontsize=29)

plt.scatter(marker_x, marker_y, 210, c=colors, edgecolors="black", linewidths=1)
plt.tight_layout()
plt.savefig(f"Q_map_{run_index}_{season}.png", dpi=150)


######################
# Full domain -- bed #
######################
fig, axes = plt.subplots(figsize=(12,5))

vel_file = "Greenland_data/BedMachineGreenland-v5_bed_smooth_sig5.nc"
r = gu.Raster(vel_file)
delta = r.res[0]*2
r.crop([-2.4e5, -2.585e6, 0, -2.47e6], inplace=True)
outline = gu.Vector(outline_path)
mask   = ~outline.create_mask(r)
r.set_mask(mask)
r.plot(cmap="terrain", cbar_title="Bed elevation")
# format
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
ax.set_ylim(-2.585e6,-2.47e6)
# domain outline
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')
# scale bar
line_x = [-2.5e5,-2.4e5]
line_y = [-2.55e6,-2.55e6]
plt.plot(line_x, line_y, color="black", lw=3)
plt.annotate("10 km", [line_x[0]-1e4,line_y[0]+5e3], c="black", fontsize=20)


# add markers and annotations
annotate_offsets = [[-1e4,1e4],[-2.7e4,-5e3],[-2.3e4,-1e4],[-2.5e4,-1e4]]
gdf_flowlines = gpd.read_file(flowlines_path)
marker_x = []
marker_y = []
for (gl,d,col,strng,offset_crd) in zip(gls,d_upglacier,colors,["(1)","(2)","(3)","(4)"],annotate_offsets):
    fl = [0,4,1,2,3][gl-1]
    coords = list(gdf_flowlines.geometry[fl].coords)
    dists = segment_lengths(gdf_flowlines.geometry[fl])
    marker_x.append(coords[np.argmin(abs(dists-d))][0])
    marker_y.append(coords[np.argmin(abs(dists-d))][1])
    ax.annotate(strng,[marker_x[-1],marker_y[-1]], xytext=[marker_x[-1]+offset_crd[0],marker_y[-1]+offset_crd[1]], c=col, fontsize=29)

plt.scatter(marker_x, marker_y, 210, c=colors, edgecolors="black", linewidths=1)
plt.tight_layout()

plt.savefig("plotting/output/B_map.jpg")


########################
# Full domain -- u_obs #
########################
plt.figure(figsize=(12,5))

vel_file = "Greenland_data/velocity/monthly/GL_vel_mosaic_Monthly_01Jul21_31Jul21_vv_v05.0.tif"
r = gu.Raster(vel_file)
delta = r.res[0]*2
r.crop([-2.4e5, -2.585e6, 0, -2.47e6], inplace=True)
outline = gu.Vector(outline_path)
mask   = ~outline.create_mask(r)
r.set_mask(mask)
r.plot(cmap=cmc.batlow, cbar_title="Observed speed (m/yr)")

# fig.colorbar(cl,  label="Surface speed (m/yr)")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
line_x = [-2.5e5,-2.4e5]
line_y = [-2.55e6,-2.55e6]
# plt.plot(line_x, line_y, color="black", lw=3)
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')
# plt.annotate("10 km", [line_x[0]-1e4,line_y[0]+5e3], c="black", fontsize=20)
ax.set_ylim(-2.585e6,-2.47e6)

# add markers and annotations
# colors = ["coral","cornflowerblue","yellowgreen"]
# annotate_offsets = [[-1e4,1e4],[-2.5e4,-5e3],[-2.3e4,-1e4]]
# gdf_flowlines = gpd.read_file(flowlines_path)
# gls = [1,3,4]
# d_upglacier = [15e3,8e3,15e3,20e3]
# marker_x = []
# marker_y = []
# for (gl,d,col,strng,offset_crd) in zip(gls,d_upglacier,colors,["(1)","(2)","(3)"],annotate_offsets):
#     fl = [0,4,1,2,3][gl-1]
#     coords = list(gdf_flowlines.geometry[fl].coords)
#     dists = segment_lengths(gdf_flowlines.geometry[fl])
#     marker_x.append(coords[np.argmin(abs(dists-d))][0])
#     marker_y.append(coords[np.argmin(abs(dists-d))][1])
#     ax.annotate(strng,[marker_x[-1],marker_y[-1]], xytext=[marker_x[-1]+offset_crd[0],marker_y[-1]+offset_crd[1]], c=col, fontsize=29)

# plt.scatter(marker_x, marker_y, 210, c=colors, edgecolors="black", linewidths=1)
plt.tight_layout()

plt.savefig("plotting/output/Uobs_map.jpg")
