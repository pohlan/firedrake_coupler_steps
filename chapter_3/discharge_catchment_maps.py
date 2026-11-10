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

run_indices = [901,903]
# model_labels  = ["sig=1", "sig=10"]
# run_indices = [479,478]
# model_labels  = ["Smaller catchment", "Reduced \nsheet flow"]
Q_min     = 15
# vmax = 5e10

flowlines_path = "Greenland_data/russel/flowlines.gpkg"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"
vel_dir = "Greenland_data/velocity/monthly/"

# plotting parameters
plt.rcParams['font.size'] = 44
annotate_fs = 44

base_cmap   = cmc.lajolla_r
cmap_colors = base_cmap(np.linspace(0.1, 1.0, 256))
capped_map_Q = colors.ListedColormap(cmap_colors)

base_cmap = cmc.lapaz
cmap_colors = base_cmap(np.linspace(0.2, 0.85, 256))
capped_map_q = colors.ListedColormap(cmap_colors)
# "BrBG" for laminar vs turbulent, Re*omega
# capped_map_q = "BrBG"

tp = f"parameter_runs/run_{run_indices[0]}/time_series.h5"
mesh_, smesh_ = get_meshes(tp)
V = df.FunctionSpace(mesh_, "CG", 1)
V_DG0 = df.FunctionSpace(mesh_, "DG", 0)
# V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
def get_topo(file):
    r = gu.Raster(file)
    return r



# hydro coords

data_dir = "/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/Greenland_data/russel/"
hydro_file = f"{data_dir}hydrological_outlets_russel.gpkg"
hydro_points = fiona.open(hydro_file, mode="r")
hydro_coords = np.array([i['geometry']['coordinates'] for i in list(hydro_points.values())])
hydro_points.close()

# glacier_names = ["Isunnguata Sermia", "Russel", "Ørkendalen", "Isorlersuup", "M.A.K."]
glacier_names = ["IS", "Russell", "Ørk", "Isorl.", "5"]
cols = cmc.berlin(np.linspace(0.0, 1.0, 5))
lw = 6

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
xc = [[xcoords[0]]]
yc = [[ycoords[0]]]
x0 = 0.5*xcoords[0] + 0.5*xcoords[1]

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



def get_Q_lines(Q_vals, ax, cax, coords):
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

# Create figure and plot
fig = plt.figure(figsize=(30, 22))
gs = fig.add_gridspec(3, 3, left=0.083, right=0.948, top=0.93, bottom=0.07, hspace=0.2, wspace=0, width_ratios=[1,1,0.12], height_ratios=[1,1,1])

# gs_bottom = gs[-1,0:2].subgridspec(1, 3, wspace=0.14, width_ratios=[0.012,1,1])

for (i_r,run_index) in enumerate(run_indices):
    # load model output
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    mesh_, smesh_ = get_meshes(timeseries_path)
    coords = smesh_.coordinates.dat.data  # mesh coordinates
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, _, n_idx, _ = load_model_output(timeseries_path)

    float_params = hlp.get_params_from_input_file(run_index)
    B, H = get_geometry(float_params["sig_topo"])

    idx = n_idx-1
    ax = fig.add_subplot(gs[0,i_r])

    # extract q for the specified timestep
    V   = df.FunctionSpace(mesh_, "CG", 1)
    q_vec_x = q_raw[::2, idx]
    q_vec_y = q_raw[1::2, idx]
    q = df.Function(V)
    q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2) / (3600*24*365)
    # q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2) / (3600*24*365) /1.787e-6 / 1000  # Re*omega

    V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
    q_vec = df.Function(V_vec)
    q_vec.dat.data[:,0] = q_raw[::2, idx]
    q_vec.dat.data[:,1] = q_raw[1::2, idx]
    n = df.FacetNormal(mesh_)

    phi = df.Function(V)
    phi.dat.data[:] = phi_raw[:,idx]
    gradphi = df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(1/2)
    keff = df.Function(V).interpolate(q / gradphi)

    labels     = get_catchments(mesh_, smesh_, phi_raw[:,idx], Q_raw[:,idx])
    labels_fct = df.Function(V)
    labels_fct.dat.data[:] = labels
    labels_DG0 = df.Function(V_DG0)
    labels_DG0.interpolate(labels_fct('+'))

    cl = tripcolor(labels_fct, axes=ax, cmap=capped_map_q) #, norm=colors.LogNorm(vmin=1e-5, vmax=0.1))
    # cl = tripcolor(B, axes=ax, cmap="terrain", vmin=-150, vmax=650) #, norm=colors.LogNorm(vmin=1e-5, vmax=0.1))
    # manually define position and size of colorbar
    cax1 = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(cl, cax=cax1, label=r"$q\,(\mathrm{m^2 s^{-1}})$")
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

    # extract Q for the specified timestep
    Q_vals = Q_raw[:, idx] / (3600*24*365)
    Q_vals[np.where(Q_vals<Q_min)] = np.nan

    cax2 = fig.add_subplot(gs[0, 2])
    lc = get_Q_lines(Q_vals, ax, cax2, coords)
    ax.add_collection(lc)
    remove_axes(ax)

    # ax.annotate(season.replace(" ", "\n"), xy=[-0.09, 0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    model_label = f"sig={int(float_params["sig_topo"])}"  # model_labels[i_r]
    ax.annotate(model_label, xy=[0.5,1.16], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    ax.annotate(f"{idx_to_letter(i_r)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.9, fontweight="bold")



    # proglacial discharge
    i_model = [n_idx-10]
    Q_termini = np.zeros((len(i_model),len(hydro_coords)))
    y_coord   = np.zeros(len(hydro_coords))
    for (i_time, i_m) in enumerate(i_model):
        # output channels
        for (i_gl, idx) in enumerate(termini_outlow_idx):
            y_coord[i_gl] = Fy.dat.data_ro[idx[0]]
            Q_termini[i_time,i_gl] = np.sum(Q_raw[idx, i_m])

    ax = fig.add_subplot(gs[1,i_r])
    idx_order = np.flip(np.argsort(y_coord))
    ax.bar(glacier_names, Q_termini[0,idx_order])
    ax.set_ylim(0,2.2e10)
    if i_r == 0:
        ax.set_ylabel("Proglacial discharge")
    print(f"Sum proglacial discharge {run_index}: {np.sum(Q_termini[0,idx_order])*1e-10} x 1e10")


    print(y_coord)
    print(idx_order)

    # catchment sizes

    size_catchments = np.zeros(5)
    q_out_terminus  = np.zeros(5)
    m_fct = df.Function(V)
    m_fct.dat.data[:] = m_raw[:,-1]
    for i in range(5):
    # for i in [4,3,2,1,0]:
        catchment = df.Function(V)
        catchment.dat.data[np.where(labels==i)] = 1
        size_catchments[4-i] = df.assemble(m_fct*catchment*df.dx)
        # q_out_terminus[4-i] = df.assemble(df.dot(q_vec, n) * catchment * df.ds(1))
        # size_catchments[i] = df.assemble(catchment*df.dx)

    ax = fig.add_subplot(gs[2,i_r])
    ax.bar(glacier_names, size_catchments)
    print(f"Sum catchments run {run_index}: {np.sum(size_catchments)*1e-10} x 1e10")
    ax.set_ylim(0,2.2e10)
    if i_r == 0:
        ax.set_ylabel("Catchment runoff")



plt.tight_layout()
# plt.savefig(f"plotting/main_figures/fS03b_{run_indices[0]}.jpg")
plt.savefig(f"chapter_3/figures/discharge2_{run_indices[0]}_{run_indices[1]}.jpg")