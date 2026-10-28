import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
from models_main.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from datetime import datetime, timedelta
import firedrake as df
from firedrake.pyplot import tripcolor
import cmcrameri.cm as cmc
import matplotlib.colors as colors
import fiona

fig_mode = ["main", "turb_vs_transition", "Re_omega", "three_catchments"][0]

draw_Q = True
if fig_mode == "main":
    run_indices = [312,706]
    model_labels  = ["Baseline", "Reduced\nsheet flow"]
    fig_height = 13
    outline_paths = ["Greenland_data/russel/russel_domain.gpkg",
                     "Greenland_data/russel/russel_domain.gpkg"]
    figname = "f03"
elif fig_mode == "turb_vs_transition":
    run_indices = [602,706]
    model_labels  = ["Turbulent", "Reduced\nsheet flow"]
    fig_height = 13
    outline_paths = ["Greenland_data/russel/russel_domain.gpkg",
                     "Greenland_data/russel/russel_domain.gpkg"]
    figname = "fS12"
elif fig_mode == "Re_omega":
    run_indices = [706]
    model_labels  = ["Reduced\nsheet flow"]
    fig_height = 8
    outline_paths = ["Greenland_data/russel/russel_domain.gpkg"]
    figname = "fS13"
    draw_Q = False
else:
    run_indices = [706,711,712]
    model_labels  = ["Red. sheet\nflow", "Smaller\ndomain", "Larger\ndomain"]
    fig_height = 19
    outline_paths = ["Greenland_data/russel/russel_domain.gpkg",
                     "Greenland_data/russel/domain_smaller.gpkg",
                     "Greenland_data/russel/domain_Mankoff_k09_5glaciers.gpkg"]
    figname = "fS20"
# run_indices = [479,478]
# model_labels  = ["Smaller catchment", "Reduced \nsheet flow"]
Q_min     = 15
# vmax = 5e10

flowlines_path = "Greenland_data/russel/flowlines.gpkg"
# outline_path   = "Greenland_data/russel/russel_domain.gpkg"
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


# V = df.FunctionSpace(mesh_, "CG", 1)
# def get_topo(file):
#     r = gu.Raster(file)
#     return r
# r_thk = get_topo("Greenland_data/BedMachineGreenland-v5_thickness_smooth_sig5.nc")
# r_bed = get_topo("Greenland_data/BedMachineGreenland-v5_bed_smooth_sig5.nc")
# v_dg = df.VectorFunctionSpace(mesh_, "CG", 1)
# X = df.assemble(df.interpolate(mesh_.coordinates,v_dg))
# meshx = X.dat.data_ro[:,0]
# meshy = X.dat.data_ro[:,1]
# B = df.Function(V)
# H = df.Function(V)
# B.dat.data[:] = r_bed.interp_points((meshx, meshy), as_array=True)
# H.dat.data[:] = r_thk.interp_points((meshx, meshy), as_array=True)

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
    # lc = mc.LineCollection(lines, cmap='Greys',
                        #    norm=plt.Normalize(vmin=np.nanmin(Q_vals)*(1), vmax=Q_min*1000))
    lc = mc.LineCollection(lines, cmap=capped_map_Q, norm=plt.Normalize(vmin=Q_min, vmax=140), joinstyle='round', capstyle='round')
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

# Create figure and plot
Ly = np.zeros(len(run_indices))
for (i,r) in enumerate(run_indices):
    tp = f"parameter_runs/run_{r}/time_series.h5"
    mesh_, smesh_ = get_meshes(tp)
    meshx, meshy = zip(*get_coordinates(mesh_, "CG", 1))
    Ly[i] = np.max(meshy) - np.min(meshy)

fig = plt.figure(figsize=(30, fig_height))
# height_ratios = Ly/Ly[0]
gs = fig.add_gridspec(len(run_indices), 3, left=0.11, right=0.948, top=0.92, bottom=0.0, hspace=-0.1, wspace=0, width_ratios=[1,1,0.12], height_ratios=Ly)

for (i_r,(run_index,outline_path)) in enumerate(zip(run_indices,outline_paths)):
    # load model output
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    mesh_, smesh_ = get_meshes(timeseries_path)
    coords = smesh_.coordinates.dat.data  # mesh coordinates
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, _, n_idx, _ = load_model_output(timeseries_path)
    for (i_s,season) in enumerate(["July 2020", "January 2021"]):
        idx = {'January 2021': int(7.05*365/2), 'July 2020': int(6.53*365/2)}[season]
        ax = fig.add_subplot(gs[i_r,i_s])

        # extract q for the specified timestep
        V   = df.FunctionSpace(mesh_, "CG", 1)
        q_vec_x = q_raw[::2, idx]
        q_vec_y = q_raw[1::2, idx]
        q = df.Function(V)
        if fig_mode == "Re_omega":
            q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2) / (3600*24*365) /1.787e-6 / 2000  # Re*omega
            cl = tripcolor(q, axes=ax, cmap=cmc.vik, norm=colors.LogNorm(vmin=1/50, vmax=50))
            clabel = r"Re $\omega$"
        else:
            q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2) / (3600*24*365)
            cl = tripcolor(q, axes=ax, cmap=capped_map_q, norm=colors.LogNorm(vmin=1e-5, vmax=0.1))
            clabel = r"$q\,(\mathrm{m^2 s^{-1}})$"

        # phi = df.Function(V)
        # phi.dat.data[:] = phi_raw[:,idx]
        # gradphi = df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(1/2)
        # keff = df.Function(V).interpolate(q / gradphi)


        # h = df.Function(V)
        # h.dat.data[:] = h_raw[:,idx]
        # h.dat.data[:] = 910*9.81*H.dat.data[:] - (phi_raw[:,idx] - 1000*9.81*B.dat.data[:])
        # h.interpolate(q / )

        # cl = tripcolor(h, axes=ax, cmap=cmc.lapaz, vmin=0, vmax=3e6) #, norm=colors.LogNorm(vmin=1e-3, vmax=3))
        # cl = tripcolor(keff, axes=ax, cmap=cmc.lapaz, norm=colors.LogNorm(vmin=1e-5, vmax=0.1))
        if (i_s == 0) & (i_r ==0):
            cax1 = fig.add_subplot(gs[0, 2])
            cbar = fig.colorbar(cl, cax=cax1, label=clabel) # cax= ... not ax=... !

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

        if draw_Q:
            # extract Q for the specified timestep
            Q_vals = Q_raw[:, idx] / (3600*24*365)
            Q_vals[np.where(Q_vals<Q_min)] = np.nan
            cax2 = fig.add_subplot(gs[1, 2])
            lc = get_Q_lines(Q_vals, ax, cax2, coords)
            ax.add_collection(lc)

        meshx, meshy = zip(*get_coordinates(mesh_, "CG", 1))
        remove_axes(ax, ylims=(np.min(meshy),np.max(meshy)))

        # if i_r == 0:
            # ax.annotate(season.replace(" ", "\n"), xy=[-0.09, 0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        if i_s == 0:
            model_label = model_labels[i_r]
            ax.annotate(model_label, xy=(-0.26, 0.5), xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", va="center")
        if i_r == 0:
            ax.annotate(season, xy=[0.5,1.15], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        ax.annotate(f"{idx_to_letter(i_r*2+i_s)}", xy=(0.03,1.05), xycoords="axes fraction", fontsize=annotate_fs*0.9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"plotting/main_figures/{figname}.jpg")
