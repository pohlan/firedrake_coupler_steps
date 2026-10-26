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
from shapely.geometry import box
import fiona
from derive_catchments_df import *

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
xc = [[xcoords[0]]]
yc = [[ycoords[0]]]
x0 = 0.5*xcoords[0] + 0.5*xcoords[1]

run_indices_turbulent = list(range(601,641))
run_indices_transition = list(range(811,851))
# run_indices_turbulent = list(range(601,611))
# run_indices_transition = list(range(811,822))
m_b_0 = 0.02
run_indices = [312,706]
model_labels  = ["Baseline", "Reduced sheet\nflow"] #, "Reduced sheet\nflow\nturbulent", "any", "any"]
cols = ["coral","cornflowerblue","yellowgreen","palevioletred"]

year0 = 2020
xstart, xend = datetime(year0,4,1),datetime(year0+1,6,1)
xstart_winter, xend_winter = datetime(year0,10,1), datetime(year0+1,4,30)
xstart_summer, xend_summer = datetime(year0,5,1), datetime(year0,9,30)

data_dir = "/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/Greenland_data/russel/"
outline_path   = f"{data_dir}russel_domain.gpkg"
gdf = gpd.read_file(outline_path)
vel_dir = "Greenland_data/velocity/monthly/"
hydro_file = f"{data_dir}hydrological_outlets_russel.gpkg"
hydro_points = fiona.open(hydro_file, mode="r")
hydro_coords = np.array([i['geometry']['coordinates'] for i in list(hydro_points.values())])

# plotting parameters
plt.rcParams['font.size'] = 25
annotate_fs = 25
lw = 3
markersz = 40

# load model input independent of run
timeseries_path = f"parameter_runs/run_477/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
coords = smesh_.coordinates.dat.data  # mesh coordinates
B, H, S = load_topography(mesh_, sig=5)
B_B = S([(xcoords[1],ycoords[1])])
print(B_B)

# get indices of outflow edges
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

print(termini_outlow_idx)

fig = plt.figure(figsize=(16, 20))
# gs  = fig.add_gridspec(3, 5, left=0.07, right=0.97, hspace=0.05, top=0.9, bottom=0.1, width_ratios=[1,0.01,0.6,0.03,0.6], height_ratios=[1,0.4,1])
gs  = fig.add_gridspec(5, 4, left=0.08, right=1.0, hspace=0.0, top=0.97, bottom=0.055, width_ratios=[1,0.08,1,0.56], height_ratios=[2.5,0.7,1,0.65,1])

gs_top = gs[0,:].subgridspec(2, 3, hspace=0.3, wspace=0.0, width_ratios=[0.1,1,0.6])
ax1 = fig.add_subplot(gs_top[0,1])
ax2 = fig.add_subplot(gs_top[1,1])
ax3 = fig.add_subplot(gs[2,0])
ax4 = fig.add_subplot(gs[2,2])
ax5 = fig.add_subplot(gs[4,0])
ax6 = fig.add_subplot(gs[4,2])

# inset for cross-section
ax_inset = ax1.inset_axes([0.45, 0.49, 0.4, 0.5])
gdf.plot(ax=ax_inset, facecolor="none", edgecolor="black", linewidth=1)
ax_inset.vlines(x0, ax_inset.get_ylim()[0]+1e4, ax_inset.get_ylim()[1]+1e4, color="grey")
ax_inset.axis("off")
ax_inset.arrow(
    x0+2e4, ax_inset.get_ylim()[0]+7e4,
    -3e4, 0,
    lw=2.5,
    head_width=6000,
    head_length=9000,
    color='black', zorder=3
)
xmin, ymin, xmax, ymax = gdf.total_bounds
clipping_box = box(x0, ymin, xmax + 1, ymax)
gdf_filtered = gdf.copy()
gdf_filtered['geometry'] = gdf.geometry.intersection(clipping_box)
gdf_filtered = gdf_filtered[~gdf_filtered.is_empty]
gdf_filtered.plot(ax=ax_inset, facecolor="none", edgecolor="black", hatch="///", alpha=0.7)


# legend with dummy plots, first for total vs channels
lax = fig.add_subplot(gs_top[0, 2])
lax.plot([], [], color='grey', linestyle="solid", label="Total \n(sheets+channels)", lw=lw)
lax.plot([], [], color='grey', linestyle="dashed", label=r"Channels", lw=lw)
# lax.plot([], [], color='grey', linestyle="dashdot", label=r"Sheets", lw=lw)
lax.legend(loc=(0.15,0.4), frameon=False, handlelength=1.4)
lax.axis("off")

# legend different colors (different model runs)
lax = fig.add_subplot(gs_top[1, 2])
lax.plot([], [], color='black', linestyle="dotted", label="Meltwater", lw=lw)
lax.plot([], [], color=cols[0], linestyle="solid", label=model_labels[0], lw=lw)
lax.plot([], [], color=cols[1], linestyle="solid", label=model_labels[1], lw=lw)
# lax.plot([], [], color=cols[2], linestyle="solid", label=model_labels[2], lw=lw)
lax.axis("off")
# add separate legend between panels
handles, labels = lax.get_legend_handles_labels()
# split into first items and last two items
handles_first = [handles[0]]
labels_first = [labels[0]]
handles_last = handles[1:]
labels_last = labels[1:]
# add first legend (no title)
first_legend = lax.legend(handles_first, labels_first, title="Input", loc=(0.15,1.2), frameon=False, alignment="left", handlelength=1.4)
lax.add_artist(first_legend)
# add second legend (with title for the last two)
lax.legend(handles_last, labels_last, title="Outflow", loc=(0.15,0.25), frameon=False, alignment="left", handlelength=1.4)
remove_axes(lax)

V   = df.FunctionSpace(mesh_, "CG", 1)
V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
V_sub = df.FunctionSpace(smesh_, "DG", 0)
x, y = df.SpatialCoordinate(mesh_)
i_upper = np.where(df.Function(V).interpolate(x).dat.data_ro > x0)[0]
upper = df.Function(V)
upper.dat.data[i_upper] = 1


# Mankoff
# from models_main.helpers import *
# gdf = gpd.read_file("Greenland_data/russel/Isunnguata_Mankoff_k_0.9.gpkg")
# meshx, meshy = zip(*get_coordinates(mesh_, "CG", 1))
# coords = mesh_.coordinates.dat.data_ro
# results = np.where(np.array([gdf.contains(Point(pt)) for pt in coords]))[0]

# IG = df.Function(V)
# IG.dat.data[results] = 1
# fig, axes = plt.subplots()
# cl = tripcolor(IG, axes=axes)
# fig.colorbar(cl)
# plt.savefig("B.jpg")


def inputs_outputs_timeseries(run_index, xstart, xend):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, S_raw, n_idx, _ = load_model_output(timeseries_path)

    phi = df.Function(V)

    dates_model, i_model = get_model_dates(n_idx, xstart, xend)
    # extract q for the specified timestep
    q_avg = np.zeros(len(i_model))
    q_out = np.zeros(len(i_model))
    q_upper = np.zeros(len(i_model))
    Q_avg = np.zeros(len(i_model))
    Q_upper = np.zeros(len(i_model))
    Q_termini = np.zeros((len(i_model),len(hydro_coords)))
    ms    = np.zeros(len(i_model))
    ms_IG = np.zeros(len(i_model))
    m_upper = np.zeros(len(i_model))
    sheet_volumes = np.zeros(len(i_model))
    channel_volumes = np.zeros(len(i_model))
    for (i_time, i_m) in enumerate(i_model):
        # input
        m     = df.Function(V)
        m.dat.data[:] = m_raw[:, i_m]
        m_in = df.assemble(m * df.dx) # m^3 / yr, melt input over full domain
        ms[i_time] = m_in
        m_upper[i_time] = df.assemble(upper * m * df.dx) # melt input over upper part
        # output sheet
        q_vec = df.Function(V_vec)
        q_vec.dat.data[:,0] = q_raw[::2, i_m]
        q_vec.dat.data[:,1] = q_raw[1::2, i_m]
        # across_q = df.assemble((df.div(q_vec)) * df.dx) # m^3 / yr
        n = df.FacetNormal(mesh_)
        q_out[i_time] = df.assemble(df.dot(q_vec, n) * df.ds(1))
        q_avg[i_time] = df.assemble(df.dot(q_vec,q_vec)**(0.5)*df.dx(domain=mesh_)) / df.assemble(df.Constant(1)*df.dx(domain=mesh_))
        q_upper[i_time] = df.assemble(upper*(df.div(q_vec)) * df.dx)
        h = df.Function(V)
        h.dat.data[:] = h_raw[:, i_m]
        sheet_volumes[i_time] = df.assemble(h*df.dx)
        # output channels
        # idx_outflow = get_outflow_index_skeleton_mesh(mesh_, smesh_)
        idx_throughflow = get_interior_flux_edges(mesh_, smesh_, x0)
        Q = df.Function(V_sub)
        Q.dat.data[:] = Q_raw[:,i_m]
        # Q_out[i_time] = np.sum(Q_raw[idx_outflow, i_m])
        Q_avg[i_time] = df.assemble(Q*df.dx(domain=smesh_)) / df.assemble(df.Constant(1)*df.dx(domain=mesh_))
        Q_upper[i_time] = np.sum(Q_raw[idx_throughflow, i_m])
        # channel size
        S = df.Function(V_sub)
        S.dat.data[:] = S_raw[:,i_m]
        channel_volumes[i_time] = df.assemble(S*df.dx(domain=smesh_))
        # channels outflow of domain
        for (i_gl, idx) in enumerate(termini_outlow_idx):
            Q_termini[i_time,i_gl] = np.sum(Q_raw[idx, i_m])
        # total budget
        # budgets[i_time] = ms[i_time] - q_out[i_time] - Q_out[i_time]

        phi.dat.data[:] = phi_raw[:,i_m]
        idx_outflow =  get_outflow_index_2D_mesh(mesh_)
        coords, cells, edges = mesh_vertex_graph(mesh_)
        values = phi.dat.data_ro.copy()
        filled, labels, parent = priority_flood_catchments(
            values,
            edges,
            idx_outflow,
        )
        IG2 = df.Function(V)
        IG2.dat.data[np.where(labels==1)] = 1
        m_IG = df.assemble(m * IG2 * df.dx) # m^3 / yr, melt input over full domain
        ms_IG[i_time] = m_IG


    return m_upper, q_upper, Q_upper, ms, q_out, Q_termini, ms_IG, dates_model, n_idx



for (i_r,(run_index,model_label,col)) in enumerate(zip(run_indices,model_labels,cols)):
    print(run_index)

    # load timeseries of fluxes total and through fluxgate (upper)
    m_upper, q_upper, Q_upper, ms, q_out, Q_termini, ms_IG, dates_model, n_idx = inputs_outputs_timeseries(run_index, xstart, xend)
    qtot_out = np.sum(Q_termini,axis=1)+q_out
    Qout_IG = Q_termini[:,0]
    qtot_out_upper = Q_upper+q_upper

    # calculate total winter discharge to compare to basal melt input
    # if run_index == 314:
    #     dates_summer, i_summer = get_model_dates(n_idx, datetime(year0,5,15), datetime(year0,10,15))
    # else:
    dates_summer, i_summer = get_model_dates(n_idx, xstart_summer, xend_summer)
    dates_winter, i_winter = get_model_dates(n_idx, xstart_winter, xend_winter)
    # i_budget_s = np.where([(d>xstart_summer) & (d<xend_summer) for d in dates_model])
    # i_budget_w = np.where([(d>xstart_winter) & (d<xend_winter) for d in dates_model])
    # budget_s = np.sum(ms[i_budget_s] - qtot_out[i_budget_s])
    # budget_s_IG = np.sum(ms_IG[i_budget_s] - Qout_IG[i_budget_s])
    # budget_w = np.sum(qtot_out[i_budget_w])
    # basal_winter = np.sum(ms[i_budget_w])
    # print(f"run index: {run_index}")
    # print(f"melt input summer: {np.sum(ms[i_budget_s])*1e-12} x 1e-12")
    # print(f"summer outflow: {np.sum(qtot_out[i_budget_s])*1e-12} x 1e-12")
    # print(f"retained water in percent: {budget_s/np.sum(ms[i_budget_s])*100}%")
    # print(f"retained water of IG in percent: {budget_s_IG/np.sum(ms_IG[i_budget_s])*100}%")
    # print(f"Total basal melt winter: {basal_winter*1e-12} x 1e-12")
    # print(f"ratio retained vs winter basal melt: {budget_s/basal_winter}")
    # print(f"retained as a percentage of retained + winter basal melt: {budget_s/(budget_s+basal_winter)}")

    # plot
    if i_r == 1:
        ax1.plot(dates_model, m_upper*1e-9/365, color="black", ls="dotted", label="Meltwater", lw=lw, zorder=1)
    ax1.plot(dates_model, qtot_out_upper*1e-9/365, label="", color=col, lw=lw, ls="solid", zorder=3)
    ax1.plot(dates_model, Q_upper*1e-9/365, label=model_label, color=col, lw=lw, ls="dashed", zorder=2)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax1.set_ylabel("Flux gate\ndischarge "+ r"$(\mathrm{km^3\,d^{-1}})$")
    y0_ax1, y1_ax1 = ax1.get_ylim()
    panel_letter_annotation(ax1, 0, annotate_fs, xyc=(0.03, 1.04))
    if i_r == 1:
        ax1.vlines(xend_winter, y0_ax1, y1_ax1*0.35, color="black", ls="-.")
        ax1.fill_betweenx([y0_ax1,y1_ax1*0.35], xstart_winter, xend_winter, facecolor="grey", alpha=0.15) # yellowgreen / darkseagreen
        ax1.set_ylim(y0_ax1,y1_ax1)

    # if i_r == 0:
    #     ax2.plot(dates_model, ms*1e-9/365, color="black", ls="dotted", label="Meltwater", lw=lw, zorder=1)
    # ax2.plot(dates_model, qtot_out*1e-9/365, label=model_label, color=col, lw=lw, ls="dashed", zorder=2)
    # ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    # ax2.set_ylabel("Discharge "+ r"$(\mathrm{km^3\,d^{-1}})$")
    # y0_ax2, y1_ax2 = ax2.get_ylim()
    # panel_letter_annotation(ax2, 0, annotate_fs, xyc=(0.03, 1.04))
    # if i_r == 1:
    #     ax2.fill_betweenx([y0_ax2,y1_ax2*0.35], xstart_winter, xend_winter, facecolor="cadetblue", alpha=0.1)
    #     ax2.set_ylim(y0_ax2,y1_ax2)

    # U_monthly = get_monthly_means_model(mesh_, run_index, xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]])
    # dU = fit_linear_slope(U_monthly)
    # print(f"dU: {dU}")

    if i_r == 1:
        model_label = "Reduced \nsheet flow"
    ax2.plot(dates_model, Q_upper*1e-9/365, label=model_label, lw=lw, color=col, ls="dashed")
    ax2.set_yscale('log')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax2.set_xlabel(f"Month of {year0} / {year0+1}")
    ax2.set_ylabel("Flux gate channel\ndischarge "+ r"$(\mathrm{km^3\,d^{-1}})$")
    if i_r == 1:
        ax2.vlines(xend_winter, -0.2, 9, color="black", ls="-.")
        ax2.annotate(r"$Q_\mathrm{\,Flux\,gate}$", xy=[datetime(year0+1,4,23), 1e-7], ha="right", rotation=90, fontsize=annotate_fs)
        y0_ax2, y1_ax2 = ax2.get_ylim()
        print([y0_ax2,y1_ax2])
        ax2.fill_betweenx([y0_ax2,y1_ax2], xstart_winter, xend_winter, facecolor="grey", alpha=0.15) # cadetblue
        ax2.annotate("Winter period", xy=[0.53, 0.85], xycoords="axes fraction", color="black", fontsize=annotate_fs, fontweight="bold")
        ax2.set_ylim(y0_ax2,0.4)
    panel_letter_annotation(ax2, 1, annotate_fs, xyc=(0.03, 1.04))


# cols_sensitivity = ["yellowgreen","cornflowerblue"]
cols_sensitivity = ["burlywood","teal"] # "" darkkhaki
flow_labels      = ["Turbulent","Transitional"]
alphas           = [1.0,1.0]
marker_types=["o", "o"]
# m_basal scaling
min_size, max_size = 40, 400  # min and max marker sizes
m_min, m_max = 0.001, 0.1
for (run_indices_suit, marker, col, alpha) in zip([run_indices_turbulent, run_indices_transition], marker_types, cols_sensitivity, alphas):
    dUs_A = []
    dUs_B = []
    # qtot_out_upper = []
    summer_net_storage = []
    m_basal = []
    Qs_channels = []
    for (i_r,run_index) in enumerate(run_indices_suit):
        print(run_index)

        timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
        # load timeseries of fluxes total and through fluxgate (upper)
        try:
            m_upper, q_upper, Q_upper, ms, q_out, Q_termini, _, dates_model, n_idx = inputs_outputs_timeseries(run_index, xstart, xend)
            if len(dates_model)==0:
                continue
        except:
            continue

        # parameters
        float_params = hlp.get_params_from_input_file(run_index)
        m_basal.append(float_params["m_basal"])
        # get winter speedup in specific locations
        # (technically loading everything again, could be improved...)
        U_monthly = get_monthly_means_model(mesh_, run_index, xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]])
        dU = fit_linear_slope(U_monthly)
        dUs_A.append(dU)
        U_monthly = get_monthly_means_model(mesh_, run_index, xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]])
        dU = fit_linear_slope(U_monthly)
        dUs_B.append(dU)

        # discharge
        i_winter = np.where([(d>xstart_winter) & (d<xend_winter) for d in dates_model])[0]
        Qs_channels.append(Q_upper[i_winter[-1]])
        # qtot_out_upper.append(np.sum(Q_upper+q_upper)/365*2)

        # water retention
        qtot_out = np.sum(Q_termini,axis=1)+q_out
        i_summer = np.where([(d>xstart_summer) & (d<xend_summer) for d in dates_model])
        budget_s = np.sum(ms[i_summer] - qtot_out[i_summer])
        summer_net_storage.append(budget_s/np.sum(ms[i_summer])*100)

    # scale m_basal to appropriate markersize array
    m_basal = np.array(m_basal)
    z_norm = (m_basal - m_min) / (m_max - m_min)
    sizes = min_size + z_norm * (max_size - min_size)
    # Point A, Q_end vs Delta u
    im = ax3.scatter(np.array(Qs_channels)*1e-9/365, dUs_A, s=sizes, c=col, marker=marker, alpha=alpha)
    # col instead of m_basal
    # Point B, Q_end vs Delta u
    im = ax4.scatter(np.array(Qs_channels)*1e-9/365, dUs_B, s=sizes, c=col, marker=marker, alpha=alpha)

    # Point A, budget vs Delta u
    # im = ax5.scatter(np.array(qtot_out_upper)*1e-9/365, dUs_A, markersz, col, marker=marker)
    im = ax5.scatter(np.array(summer_net_storage), dUs_A, s=sizes, c=col, marker=marker, alpha=alpha)

    # Point A, budget vs Delta u
    # im = ax6.scatter(np.array(qtot_out_upper)*1e-9/365, dUs_B, markersz, col, marker=marker)
    im = ax6.scatter(np.array(summer_net_storage), dUs_B, s=sizes, c=col, marker=marker, alpha=alpha)

    # # shared colorbar --> make it a legend! (and replace cmap with color list in scatter plots)
    # cax = fig.add_subplot(gs[2,3])
    # fig.colorbar(im, cax=cax, label=r"$m_\mathrm{basal}$ ($\mathrm{m\,a^{-1}}$)")
    # # manually define position and size of colorbar
    # pos = cax.get_position()
    # cax.set_position([
    #     pos.x0 + 0.1 * pos.width,
    #     pos.y0 - 0.8 * pos.height,
    #     0.07*pos.width,
    #     pos.height
    # ])


ax3.set_xlabel(r"$Q_\mathrm{\,Flux\,gate}$ ($\mathrm{km^3\,d^{-1}}$)")
ax3.set_ylabel(r"$\dot{u}_\mathrm{winter}$ ($\mathrm{m\,a^{-1}}$/month)")
panel_letter_annotation(ax3, 2, annotate_fs, xyc=(0.03, 1.04))
ax3.set_xscale('log')
ax3.set_title("Point A")

ax4.set_xlabel(r"$Q_\mathrm{\,Flux\,gate}$ ($\mathrm{km^3\,d^{-1}}$)")
ax4.set_ylabel(r"$\dot{u}_\mathrm{winter}$ ($\mathrm{m\,a^{-1}}$/month)")
panel_letter_annotation(ax4, 3, annotate_fs, xyc=(0.03, 1.04))
ax4.set_xscale('log')
ax4.set_title("Point B")
y0, y1 = ax4.get_ylim()
ax4.set_ylim(y0, y1+0.3)

ax5.set_xlabel(r"Summer retention (%)")
ax5.set_ylabel(r"$\dot{u}_\mathrm{winter}$ ($\mathrm{m\,a^{-1}}$/month)")
panel_letter_annotation(ax5, 4, annotate_fs, xyc=(0.03, 1.04))
# ax5.set_xscale('log')
ax5.set_title("Point A")

ax6.set_xlabel(r"Summer retention (%)")
ax6.set_ylabel(r"$\dot{u}_\mathrm{winter}$ ($\mathrm{m\,a^{-1}}$/month)")
panel_letter_annotation(ax6, 5, annotate_fs, xyc=(0.03, 1.04))
# ax6.set_xscale('log')
ax6.set_title("Point B")
ax6.set_ylim(y0, y1+0.3)

# legend with dummy plots, first for total vs channels
m_basal = np.array([0.001,0.02,0.05,0.1])
z_norm = (m_basal - m_min) / (m_max - m_min)
sizes = min_size + z_norm * (max_size - min_size)
lax = fig.add_subplot(gs[2,3])
for (m,s) in zip(m_basal,sizes):
    lax.scatter([], [], s=s, c="grey", marker="o", label=f"{m}")
lax.legend(loc=(-0.1,-0.1), frameon=False, title=r"$m_\mathrm{basal}$ ($\mathrm{m\,a^{-1}}$)")
lax.axis("off")



# legend with dummy plots, first for total vs channels
lax = fig.add_subplot(gs[4,3])
for (c,lab,alpha) in zip(cols_sensitivity,flow_labels,alphas):
    lax.scatter([], [], s=np.mean(sizes), c=c, alpha=alpha, marker="o", label=lab)
lax.legend(loc=(-0.2,0.7), frameon=False, title="Sheet flow")
lax.axis("off")

plt.savefig(f"plotting/main_figures/f06.jpg")
