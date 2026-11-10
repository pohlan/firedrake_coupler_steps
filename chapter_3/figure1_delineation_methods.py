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
import fiona
from get_catchments_Q import *
import models_main.helpers as hlp
from test_polygonize import *

run_index = 906
Q_min     = 15

# outline_path   = "Greenland_data/russel/russel_domain.gpkg"
outline_path   = "Greenland_data/russel/russel_domain_6_glaciers.gpkg"
gdf = gpd.read_file(outline_path)
catchments_Mankoff = "Greenland_data/russel/basins_Mankoff_k_0.9.gpkg"

mankoff_indices = [[27538,], # Isunnguata Sermia
                   [27597,27609] # Russell glacier
                    ]

# plotting parameters
plt.rcParams['font.size'] = 44
annotate_fs = 44



# cmap = plt.get_cmap("berlin", 6)
# color_map = {
#     i: cmap(i)
#     for i in range(6)
# }

# base_cmap = cmc.lisbon
base_cmap = plt.get_cmap("tab20c")
cmap_colors = np.array(base_cmap.colors)[[5,2,1,0]]
color_map = {
    i: cmap_colors[i-1]
    for i in range(1,5)
}

# hydro coords
data_dir = "/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/Greenland_data/russel/"
# hydro_file = f"{data_dir}hydrological_outlets_russel.gpkg"
hydro_file = f"{data_dir}hydrological_outlets_6_glaciers.gpkg"
hydro_points = fiona.open(hydro_file, mode="r")
hydro_coords = np.array([i['geometry']['coordinates'] for i in list(hydro_points.values())])
hydro_points.close()

# glacier_names = ["Isunnguata Sermia", "Russel", "Ørkendalen", "Isorlersuup", "M.A.K."]
glacier_names = ["N","IS", "Russell", "Ørk", "Isorl.", "5"]
cols = cmc.berlin(np.linspace(0.0, 1.0, 5))
lw = 6

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
xc = [[xcoords[0]]]
yc = [[ycoords[0]]]
x0 = 0.5*xcoords[0] + 0.5*xcoords[1]

def get_Q_lines(Q_vals, ax, coords, cax=None):
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
    lc = mc.LineCollection(lines, cmap='Greys_r', norm=plt.Normalize(vmin=np.nanmin(Q_vals)*(1), vmax=Q_min*1000), joinstyle='round', capstyle='round')
    # lc = mc.LineCollection(lines, cmap=capped_map_Q, norm=plt.Normalize(vmin=Q_min, vmax=140), joinstyle='round', capstyle='round')
                        #    norm=plt.Normalize(vmin=np.nanmin(Q_vals)*(1), vmax=Q_min))
    lc.set_array(np.array(colors))
    lc.set_linewidth(3)

    if cax is not None:
        cbar = fig.colorbar(lc, cax=cax, label=r"$Q\,(\mathrm{m^3 s^{-1}})$")
        # manually define position and size of colorbar
        pos = cax.get_position()
        cax.set_position([
            pos.x0 + 0.2 * pos.width,
            pos.y0 + 0.25 * pos.height,
            0.2*pos.width,
            0.5*pos.height
        ])

    return lc

def get_topo(file):
    r = gu.Raster(file)
    return r

def get_geometry(sig):
    r_thk = get_topo(f"Greenland_data/BedMachineGreenland-v5_thickness_smooth_sig{int(sig)}.nc")
    r_bed = get_topo(f"Greenland_data/BedMachineGreenland-v5_bed_smooth_sig{int(sig)}.nc")
    v_dg = df.VectorFunctionSpace(mesh_, "CG", 1)
    X = df.assemble(df.interpolate(mesh_.coordinates,v_dg))
    meshx = X.dat.data_ro[:,0]
    meshy = X.dat.data_ro[:,1]
    B = df.Function(V)
    H = df.Function(V)
    B.dat.data[:] = r_bed.interp_points((meshx, meshy), as_array=True)
    H.dat.data[:] = r_thk.interp_points((meshx, meshy), as_array=True)
    return B, H

def westernmost_vertex_y(geom):
    if geom.geom_type == "Polygon":
        polygons_list = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons_list = list(geom.geoms)
    else:
        return np.nan
    vertices = [
        point
        for poly in polygons_list
        for point in poly.exterior.coords
    ]
    westernmost_x = min(p[0] for p in vertices)
    # If multiple vertices share the smallest x, take the northernmost
    return max(p[1] for p in vertices if p[0] == westernmost_x)


def plot_catchments(ax, labels):
    # fct_to_plot = df.Function(V_DG0)
    # fct_to_plot.dat.data[:] = labels_DG0.dat.data_ro
    coords_mesh = mesh_.coordinates.dat.data
    assert len(labels) == coords_mesh.shape[0]
    # values = labels[labels != -1]
    # coords_mesh = coords_mesh[labels != -1]
    # print(values.shape)
    # Generate polygons
    polygons = polygonize_domain_smooth(coordinates=coords_mesh,values=labels,domain=gdf.geometry.iloc[0])
    # Calculate average y-coordinate (centroid)
    polygons["sort_y"] = polygons.geometry.apply(westernmost_vertex_y)
    # Sort from north to south and assign values 0, 1, 2, ...
    polygons = polygons.sort_values(by="sort_y",ascending=False).reset_index(drop=True)
    polygons["value"] = range(len(polygons))
    polygons_filtered = polygons[(polygons["value"] > 0) & (polygons["value"] < 5)]
    # unique_values = sorted(polygons_filtered["value"].unique())
    polygons_filtered.plot(ax=ax, color=polygons_filtered["value"].map(color_map),edgecolor="black",linewidth=0.3) #,alpha=0.8)
    gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Domain')

def get_catchment_melt(labels):
    Fx_mesh   = df.Function(V).interpolate(x)
    Fy_mesh   = df.Function(V).interpolate(y)
    m_fct = df.Function(V)
    m_fct.dat.data[:] = m_raw[:,-1]
    unique_labels = np.unique(labels)
    labels_fct = df.Function(V)
    labels_fct.dat.data[:] = labels
    melt_catchments = np.zeros(len(unique_labels))
    y_coord = np.zeros(len(unique_labels))
    for i in range(len(unique_labels)):
        catchment = df.Function(V)
        i_catchment = np.where(labels_fct.dat.data_ro==i)[0]
        catchment.dat.data[i_catchment] = 1
        melt_catchments[i] = df.assemble(m_fct*catchment*df.dx)
        ix = np.argmin(Fx_mesh.dat.data_ro[i_catchment])
        y_coord[i] = Fy_mesh.dat.data_ro[i_catchment][ix]
    idx_order = np.flip(np.argsort(y_coord))
    return melt_catchments[idx_order] # catchments_all

def get_catchment_from_polygons(polygons):
    coords_mesh = mesh_.coordinates.dat.data
    m_fct = df.Function(V)
    m_fct.dat.data[:] = m_raw[:,-1]
    melt_catchments = np.zeros(len(polygons))
    for (i,polys) in enumerate(polygons):
        merged_polygons = polys.geometry.union_all()
        i_catchment = np.array([merged_polygons.covers(Point(x, y)) for x, y in coords_mesh])
        catchment = df.Function(V)
        catchment.dat.data[i_catchment] = 1
        melt_catchments[i] = df.assemble(m_fct*catchment*df.dx)
    return melt_catchments


def plot_channels(ax, Q_raw):
    Q_vals = Q_raw[:, -1] / (3600*24*365)
    Q_vals[np.where(Q_vals<Q_min)] = np.nan
    lc = get_Q_lines(Q_vals, ax, coords)
    ax.add_collection(lc)
    remove_axes(ax)


def get_DG0_interpolated_labels(mesh, labels):
    labels_fct = df.Function(V)
    labels_fct.dat.data[:] = labels


    # # Input CG1 function
    # V = df.FunctionSpace(mesh, "CG", 1)
    # label_fct = df.Function(V)
    # label_fct.dat.data[:] = labels

    unique_labels = np.unique(labels_fct.dat.data_ro)
    # Output DG0 function
    W = df.FunctionSpace(mesh, "DG", 0)
    labels_DG0 = df.Function(W)

    # Get CG1 nodal values associated with each cell
    cell_nodes = V.cell_node_list
    nodal_values = labels_fct.dat.data_ro_with_halos

    cell_labels = nodal_values[cell_nodes]

    # Count occurrences of each terrain label
    counts = np.stack([
        np.sum(cell_labels == i, axis=1)
        for i in unique_labels
    ], axis=1)

    # Select the most common label
    # np.argmax resolves ties in favour of the lowest label
    majority_labels = unique_labels[np.argmax(counts, axis=1)]

    # Assign to DG0 function
    labels_DG0.dat.data[:] = majority_labels

    unique_labels = np.unique(labels_DG0.dat.data_ro)
    return labels_DG0


# Create figure and plot
fig = plt.figure(figsize=(30, 13))
gs = fig.add_gridspec(2, 2, left=0.083, right=0.948, top=0.99, bottom=0.07, hspace=0.2, wspace=0, height_ratios=[1,0.5])


# get geometry and model output
timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
coords = smesh_.coordinates.dat.data  # mesh coordinates
float_params = hlp.get_params_from_input_file(run_index)
sig = float_params["sig_topo"]
us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, _, n_idx, _ = load_model_output(timeseries_path)
V = df.FunctionSpace(mesh_, "CG", 1)
B, H = get_geometry(sig)
V_DG0 = df.FunctionSpace(mesh_, "DG", 0)


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
df.VTKFile("Fout.pvd").write(Fout)




# 1) calculate catchments from p_w = k*p_i
# k   = 1.0
# phi = df.Function(V)
# rho_i = 910
# g     = 9.81
# rho_w = 1000
# phi.interpolate(k*rho_i*g*H + rho_w*g*B)
# labels     = get_catchments(mesh_, smesh_, phi.dat.data_ro, method="steepest")
ax = fig.add_subplot(gs[0,0])
# labels_DG0 = get_DG0_interpolated_labels(mesh_, labels)
# plot_catchments(ax, labels)
gg_Mank = gpd.read_file(catchments_Mankoff)
gg_Mank.to_crs(gdf.crs, inplace=True)
gg_polygons = []
for (i,indices) in enumerate(mankoff_indices):
    gg = gg_Mank.loc[indices].copy()
    gg["value"] = i+1
    clipped = gpd.clip(gg, gdf)
    clipped.plot(ax=ax, edgecolor="black", color=clipped["value"].map(color_map), linewidth=1.0)
    gg_polygons.append(clipped)
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Domain')
remove_axes(ax)
# melt per catchment
melt_catchments = get_catchment_from_polygons(gg_polygons)
ax = fig.add_subplot(gs[1,0])
ax.bar(glacier_names[1:3], melt_catchments, color=cmap_colors)
# # print(f"Sum catchments run {run_index}: {np.sum(size_catchments)*1e-10} x 1e10")
ax.set_ylim(0,2.2e10)
ax.set_ylabel("Catchment runoff")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# # 2) calculate catchments from modeled phi but ignoring Q
# labels     = get_catchments(mesh_, smesh_, phi_raw[:,-1], Q_raw[:,-1], method="steepest")
# ax = fig.add_subplot(gs[0,1])
# labels_DG0 = get_DG0_interpolated_labels(mesh_, labels)
# plot_catchments(ax, labels)
# # plot channels
# plot_channels(ax, Q_raw)
# # melt per catchment
# melt_catchments = get_catchment_melt(labels_DG0)
# ax = fig.add_subplot(gs[1,1])
# ax.bar(glacier_names, melt_catchments)
# ax.set_ylim(0,2.2e10)

# 2) calculate catchments from modeled phi using Q
labels     = get_catchments(mesh_, smesh_, phi_raw[:,-1], Q_raw[:,-1], method="Q")
ax = fig.add_subplot(gs[0,1])
# labels_DG0 = get_DG0_interpolated_labels(mesh_, labels)
plot_catchments(ax, labels)
# plot channels
plot_channels(ax, Q_raw)
# melt per catchment
melt_catchments = get_catchment_melt(labels)

ax = fig.add_subplot(gs[1,1])
ax.bar(glacier_names[1:-1], melt_catchments[1:-1], color=cmap_colors)
ax.set_ylim(0,2.2e10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# ax.annotate(model_label, xy=[0.5,1.16], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
# ax.annotate(f"{idx_to_letter(i_r)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.9, fontweight="bold")



# for (i_r,run_index) in enumerate(run_indices):
#     # load model output
#     timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
#     mesh_, smesh_ = get_meshes(timeseries_path)
#     coords = smesh_.coordinates.dat.data  # mesh coordinates
#     us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, _, n_idx, _ = load_model_output(timeseries_path)



#     idx = n_idx-1
#     ax = fig.add_subplot(gs[0,i_r])

#     # extract q for the specified timestep
#     V   = df.FunctionSpace(mesh_, "CG", 1)
#     q_vec_x = q_raw[::2, idx]
#     q_vec_y = q_raw[1::2, idx]
#     q = df.Function(V)
#     q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2) / (3600*24*365)
#     # q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2) / (3600*24*365) /1.787e-6 / 1000  # Re*omega

#     V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
#     q_vec = df.Function(V_vec)
#     q_vec.dat.data[:,0] = q_raw[::2, idx]
#     q_vec.dat.data[:,1] = q_raw[1::2, idx]
#     n = df.FacetNormal(mesh_)

#     phi = df.Function(V)
#     phi.dat.data[:] = phi_raw[:,idx]
#     gradphi = df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(1/2)
#     keff = df.Function(V).interpolate(q / gradphi)

#     labels     = get_catchments(mesh_, smesh_, phi_raw[:,idx], Q_raw[:,idx])
#     labels_fct = df.Function(V)
#     labels_fct.dat.data[:] = labels
#     labels_DG0 = df.Function(V_DG0)
#     labels_DG0.interpolate(labels_fct('+'))

#     cl = tripcolor(labels_fct, axes=ax, cmap=capped_map_q) #, norm=colors.LogNorm(vmin=1e-5, vmax=0.1))
#     # cl = tripcolor(B, axes=ax, cmap="terrain", vmin=-150, vmax=650) #, norm=colors.LogNorm(vmin=1e-5, vmax=0.1))
#     # manually define position and size of colorbar
#     cax1 = fig.add_subplot(gs[1, 2])
#     cbar = fig.colorbar(cl, cax=cax1, label=r"$q\,(\mathrm{m^2 s^{-1}})$")
#     pos = cax1.get_position()
#     cax1.set_position([
#         pos.x0 + 0.2 * pos.width,
#         pos.y0 + 0.25 * pos.height,
#         0.2*pos.width,
#         0.5*pos.height
#     ])

#     # domain outline
#     gdf = gpd.read_file(outline_path)
#     gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Domain')

#     # extract Q for the specified timestep
#     Q_vals = Q_raw[:, idx] / (3600*24*365)
#     Q_vals[np.where(Q_vals<Q_min)] = np.nan

#     cax2 = fig.add_subplot(gs[0, 2])
#     lc = get_Q_lines(Q_vals, ax, cax2, coords)
#     ax.add_collection(lc)
#     remove_axes(ax)

#     # ax.annotate(season.replace(" ", "\n"), xy=[-0.09, 0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
#     model_label = f"sig={int(float_params["sig_topo"])}"  # model_labels[i_r]
#     ax.annotate(model_label, xy=[0.5,1.16], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
#     ax.annotate(f"{idx_to_letter(i_r)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.9, fontweight="bold")



#     # proglacial discharge
#     i_model = [n_idx-10]
#     Q_termini = np.zeros((len(i_model),len(hydro_coords)))
#     y_coord   = np.zeros(len(hydro_coords))
#     for (i_time, i_m) in enumerate(i_model):
#         # output channels
#         for (i_gl, idx) in enumerate(termini_outlow_idx):
#             y_coord[i_gl] = Fy.dat.data_ro[idx[0]]
#             Q_termini[i_time,i_gl] = np.sum(Q_raw[idx, i_m])

#     ax = fig.add_subplot(gs[1,i_r])
#     idx_order = np.flip(np.argsort(y_coord))
#     ax.bar(glacier_names, Q_termini[0,idx_order])
#     ax.set_ylim(0,2.2e10)
#     if i_r == 0:
#         ax.set_ylabel("Proglacial discharge")
#     print(f"Sum proglacial discharge {run_index}: {np.sum(Q_termini[0,idx_order])*1e-10} x 1e10")


#     print(y_coord)
#     print(idx_order)

#     # catchment sizes

#     # size_catchments = np.zeros(5)
#     # q_out_terminus  = np.zeros(5)
#     # m_fct = df.Function(V)
#     # m_fct.dat.data[:] = m_raw[:,-1]
#     # for i in range(5):
#     # # for i in [4,3,2,1,0]:
#     #     catchment = df.Function(V)
#     #     catchment.dat.data[np.where(labels==i)] = 1
#     #     size_catchments[4-i] = df.assemble(m_fct*catchment*df.dx)
#     #     # q_out_terminus[4-i] = df.assemble(df.dot(q_vec, n) * catchment * df.ds(1))
#     #     # size_catchments[i] = df.assemble(catchment*df.dx)

#     ax = fig.add_subplot(gs[2,i_r])
#     ax.bar(glacier_names, size_catchments)
#     print(f"Sum catchments run {run_index}: {np.sum(size_catchments)*1e-10} x 1e10")
#     ax.set_ylim(0,2.2e10)
#     if i_r == 0:
#         ax.set_ylabel("Catchment runoff")



plt.tight_layout()
# plt.savefig(f"plotting/main_figures/fS03b_{run_indices[0]}.jpg")
plt.savefig(f"chapter_3/figures/f01_delineation_method.jpg")