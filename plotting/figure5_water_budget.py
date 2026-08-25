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

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
xc = [[xcoords[0]]]
yc = [[ycoords[0]]]
x0 = 0.5*xcoords[0] + 0.5*xcoords[1]

run_indices_suit = list(range(601,725))
m_b_0 = 0.02
run_indices = [314,478,656]
model_labels  = ["Baseline", "Reduced sheet \nflow", r"$h_r=8$ m", "any", "any"]
cols = ["coral","cornflowerblue","yellowgreen","palevioletred"]

xstart, xend = datetime(2020,4,1),datetime(2021,6,1)
# xstart,xend = datetime(2018,5,1),datetime(2022,6,1)
xstart_winter, xend_winter = datetime(2020,10,1), datetime(2021,4,30)

outline_path   = "Greenland_data/russel/russel_domain.gpkg"
gdf = gpd.read_file(outline_path)
vel_dir = "Greenland_data/velocity/monthly/"

# plotting parameters
plt.rcParams['font.size'] = 19
annotate_fs = 24
lw = 2.5
markersz = 30

# load model input independent of run
timeseries_path = f"parameter_runs/run_477/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
coords = smesh_.coordinates.dat.data  # mesh coordinates
B, H, S = load_topography(mesh_, sig=5)

fig = plt.figure(figsize=(20, 10))
gs  = fig.add_gridspec(3, 3, left=0.07, right=0.95, hspace=0.05, top=0.9, bottom=0.1, width_ratios=[1,0.01,0.6], height_ratios=[1,0.4,1])

# gs_top = gs[0].subgridspec(1, 5, hspace=0.0, wspace=0.04, )
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[2,0])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[2,2])

# inset for cross-section
ax_inset = ax1.inset_axes([0.02, 0.5, 0.21, 0.5])
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


# legend with dummy plots
lax = fig.add_subplot(gs[1, 0])
lax.plot([], [], color='grey', linestyle="solid", label=r"Sheets", lw=lw)
lax.plot([], [], color='grey', linestyle="dashed", label=r"Channels", lw=lw)
lax.plot([], [], color='grey', linestyle="dashdot", label=r"Sheets+channels", lw=lw)
lax.legend(loc=(0.1,0.2), frameon=False, ncol=3)
lax.axis("off")

for (i_r,(run_index,model_label,col)) in enumerate(zip(run_indices,model_labels,cols)):
    print(run_index)
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, S_raw, n_idx, _ = load_model_output(timeseries_path)
    dates_model, i_model = get_model_dates(n_idx, xstart, xend)

    # extract q for the specified timestep
    q_avg = np.zeros(len(i_model))
    q_out = np.zeros(len(i_model))
    q_upper = np.zeros(len(i_model))
    Q_avg = np.zeros(len(i_model))
    Q_out = np.zeros(len(i_model))
    Q_upper = np.zeros(len(i_model))
    ms    = np.zeros(len(i_model))
    m_upper = np.zeros(len(i_model))
    sheet_volumes = np.zeros(len(i_model))
    channel_volumes = np.zeros(len(i_model))
    V   = df.FunctionSpace(mesh_, "CG", 1)
    V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
    V_sub = df.FunctionSpace(smesh_, "DG", 0)
    x, y = df.SpatialCoordinate(mesh_)
    i_upper = np.where(df.Function(V).interpolate(x).dat.data_ro > x0)[0]
    upper = df.Function(V)
    upper.dat.data[i_upper] = 1
    for (i_time, i_m) in enumerate(i_model):
        # input
        m     = df.Function(V)
        m.dat.data[:] = m_raw[:, i_m]
        m_in = df.assemble(m * df.dx) # m^3 / yr
        ms[i_time] = m_in
        m_upper[i_time] = df.assemble(upper * m * df.dx)
        # output sheet
        q_vec = df.Function(V_vec)
        q_vec_x = q_raw[::2, i_m]
        q_vec_y = q_raw[1::2, i_m]
        q_vec.dat.data[:,0] = q_vec_x
        q_vec.dat.data[:,1] = q_vec_y
        # across_q = df.assemble((df.div(q_vec)) * df.dx) # m^3 / yr
        n = df.FacetNormal(mesh_)
        q_out[i_time] = df.assemble(df.dot(q_vec, n) * df.ds(1))
        q_avg[i_time] = df.assemble(df.dot(q_vec,q_vec)**(0.5)*df.dx(domain=mesh_)) / df.assemble(df.Constant(1)*df.dx(domain=mesh_))
        q_upper[i_time] = df.assemble(upper*(df.div(q_vec)) * df.dx)
        h = df.Function(V)
        h.dat.data[:] = h_raw[:, i_m]
        sheet_volumes[i_time] = df.assemble(h*df.dx)
        # output channels
        idx_outflow = get_outflow_index_skeleton_mesh(mesh_, smesh_)
        idx_throughflow = get_interior_flux_edges(mesh_, smesh_, x0)
        Q = df.Function(V_sub)
        Q.dat.data[:] = Q_raw[:,i_m]
        Q_out[i_time] = np.sum(Q_raw[idx_outflow, i_m])
        Q_avg[i_time] = df.assemble(Q*df.dx(domain=smesh_)) / df.assemble(df.Constant(1)*df.dx(domain=mesh_))
        Q_upper[i_time] = np.sum(Q_raw[idx_throughflow, i_m])
        # channel size
        S = df.Function(V_sub)
        S.dat.data[:] = S_raw[:,i_m]
        channel_volumes[i_time] = df.assemble(S*df.dx(domain=smesh_))

    qtot_out = Q_out+q_out
    qtot_out_upper = Q_upper+q_upper

    if i_r == 0:
        ax1.plot(dates_model, m_upper/(365*24*3600), color="black", ls="dotted", label="Meltwater", lw=lw)
    ax1.plot(dates_model, q_upper/(365*24*3600), label=model_label, color=col, lw=lw)
    ax1.fill_between(dates_model, q_upper/(365*24*3600), qtot_out_upper/(365*24*3600), alpha=0.3, facecolor=col)
    ax1.plot(dates_model, qtot_out_upper/(365*24*3600), label="", color=col, lw=lw, ls="dashdot", alpha=0.5)
    ax1.legend()
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax1.set_ylabel("Discharge "+ r"$(\mathrm{m^3\,a^{-1}})$")
    # ax1.set_yscale("log")
    ax1.annotate("Upper domain: melt input vs. outflow through cross-section", xy=(0.45, 1.2), xycoords="axes fraction", fontsize=annotate_fs)
    panel_letter_annotation(ax1, 0, annotate_fs, xyc=(0.03, 1.04))

    # add separate legend between panels
    handles, labels = ax1.get_legend_handles_labels()
    # split into first items and last two items
    handles_first = [handles[0]]
    labels_first = [labels[0]]
    handles_last = handles[1:]
    labels_last = labels[1:]
    # add first legend (no title)
    first_legend = ax1.legend(handles_first, labels_first, title="Input", loc="upper center", frameon=False)
    ax1.add_artist(first_legend)
    # add second legend (with title for the last two)
    ax1.legend(handles_last, labels_last, title="Outflow", loc="upper right", frameon=False)


    if i_r == 1:
        model_label = "Reduced \nsheet flow"
    ax2.plot(dates_model, Q_upper/(365*24*3600), label=model_label, lw=lw, color=col, ls="dashed")
    ax2.set_yscale('log')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax2.set_xlabel("Month of 2020 / 2021")
    ax2.set_ylabel("Channel discharge "+ r"$(\mathrm{m^3\,a^{-1}})$")
    if i_r == 1:
        ax2.vlines(xend_winter, -0.2, 100, color="black", ls="-.")
        ax2.annotate(r"$Q_\mathrm{\,end}$", xy=[datetime(2021,5,3), 0.6]) #, xycoords="axes fraction")
    panel_letter_annotation(ax2, 2, annotate_fs, xyc=(0.03, 1.04))



dUs_A = []
dUs_B = []
budget_tot = []
m_basal = []
alpha_s = []
h_r     = []
V_sheet_max = []
V_sheet_sum = []
V_channels_sum = []
qs_sheet_rel = []
Qs_channels = []
for (i_r,run_index) in enumerate(run_indices_suit):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    try:
        us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, S_raw, n_idx, _ = load_model_output(timeseries_path)
        dates_model, i_model = get_model_dates(n_idx, xstart_winter, xend_winter)
        if len(dates_model)==0:
            continue
    except:
        continue

    print(run_index)

    # extract q for the specified timestep
    q_out    = np.zeros(len(i_model))
    q_avg    = np.zeros(len(i_model))
    Q_out    = np.zeros(len(i_model))
    Q_avg = np.zeros(len(i_model))
    Q_upper = np.zeros(len(i_model))
    ms    = np.zeros(len(i_model))
    budgets = np.zeros(len(i_model))
    sheet_volumes = np.zeros(len(i_model))
    channel_volumes = np.zeros(len(i_model))
    V   = df.FunctionSpace(mesh_, "CG", 1)
    V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
    V_DG0 = df.FunctionSpace(smesh_, "DG", 0)
    for (i_time, i_m) in enumerate(i_model):
        # input
        m     = df.Function(V)
        m.dat.data[:] = m_raw[:, i_m] #- m_b
        ms[i_time] = df.assemble(m * df.dx) # m^3 / yr
        # output sheet
        q_vec = df.Function(V_vec)
        q_vec_x = q_raw[::2, i_m]
        q_vec_y = q_raw[1::2, i_m]
        q_vec.dat.data[:,0] = q_vec_x
        q_vec.dat.data[:,1] = q_vec_y
        q_out[i_time] = df.assemble(df.dot(q_vec, n) * df.ds(1))
        q_avg[i_time] = df.assemble(df.dot(q_vec,q_vec)**(0.5)*df.dx(domain=mesh_)) / df.assemble(df.Constant(1)*df.dx(domain=mesh_))
        # sheet volume
        h = df.Function(V)
        h.dat.data[:] = h_raw[:, i_m]
        sheet_volumes[i_time] = df.assemble(h*df.dx)
        # output channels
        idx_outflow = get_outflow_index_skeleton_mesh(mesh_, smesh_)
        idx_throughflow = get_interior_flux_edges(mesh_, smesh_, x0)
        Q = df.Function(V_sub)
        Q.dat.data[:] = Q_raw[:,i_m]
        Q_out[i_time] = np.sum(Q_raw[idx_outflow, i_m])
        Q_avg[i_time] = df.assemble(Q*df.dx(domain=smesh_)) / df.assemble(df.Constant(1)*df.dx(domain=smesh_))
        Q_upper[i_time] = np.sum(Q_raw[idx_throughflow, i_m])
        # channel size
        S = df.Function(V_sub)
        S.dat.data[:] = S_raw[:,i_m]
        channel_volumes[i_time] = df.assemble(S*df.dx(domain=smesh_))
        # total budget
        budgets[i_time] = ms[i_time] - q_out[i_time] - Q_out[i_time]

    budget_tot.append(np.sum(budgets))

    # parameters
    float_params = hlp.get_params_from_input_file(run_index)
    m_basal.append(float_params["m_basal"])
    alpha_s.append(float_params["alpha_s"])
    h_r.append(float_params["h_r"])

    # get winter speedup in specific locatiodUsn
    U_monthly = get_monthly_means_model(mesh_, run_index, xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]])
    dU = fit_linear_slope(U_monthly)
    dUs_A.append(dU)
    U_monthly = get_monthly_means_model(mesh_, run_index, xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]])
    dU = fit_linear_slope(U_monthly)
    dUs_B.append(dU)

    # discharge
    Qs_channels.append(Q_upper[-1])

    # volume
    V_sheet_sum.append((sheet_volumes/(sheet_volumes+channel_volumes))[-1])
    # i_hmax = np.argmax(np.array(sheet_volumes))
    # V_sheet_sum.append(sheet_volumes[-1]) #/sheet_volumes[i_hmax])
    V_channels_sum.append(channel_volumes[-1])
    # V_sheet_sum.append(np.sum(np.array(sheet_volumes)))

    xc_input, yc_input = prepare_location_inputs(xc, yc)
    h_winter_loc = load_model_timeseries(mesh_, h_raw, xc_input, yc_input, i_model)[:, 0]
    V_sheet_max.append(h_winter_loc.mean())


im = ax3.scatter(np.array(Qs_channels)/(3600*24*365), dUs_A, markersz, m_basal, marker="o", cmap=cmc.berlin)
ax3.set_ylabel(r"$\Delta u$ " + r"($\mathrm{m\,a^{-1}}$)")
panel_letter_annotation(ax3, 1, annotate_fs, xyc=(0.03, 1.04))
ax3.set_xscale('log')
ax3.set_title("Point A")
fig.colorbar(im, ax=ax3, label=r"$m_\mathrm{basal}$ ($\mathrm{m\,a^{-1}}$)")

im = ax4.scatter(np.array(Qs_channels)/(3600*24*365), dUs_B, markersz, m_basal, marker="o", cmap=cmc.berlin)
ax4.set_xlabel(r"$Q_\mathrm{\,end}$ ($\mathrm{m^3\,a^{-1}}$)")
ax4.set_ylabel(r"$\Delta u$ " + r"($\mathrm{m\,a^{-1}}$)")
panel_letter_annotation(ax4, 3, annotate_fs, xyc=(0.03, 1.04))
ax4.set_xscale('log')
ax4.set_title("Point B")
fig.colorbar(im, ax=ax4, label=r"$m_\mathrm{basal}$ ($\mathrm{m\,a^{-1}}$)")

plt.savefig(f"plotting/main_figures/f05.jpg")
