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

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
xc = [[xcoords[0]]]
yc = [[ycoords[0]]]

run_indices_suit = list(range(601,706))
m_b_0 = 0.02
run_indices = [327,477]
# run_indices = [478,652,705]
model_labels  = ["Baseline", "Reduced sheet \nflow", "any", "any", "any"]
cols = ["coral","cornflowerblue","yellowgreen","palevioletred"]

xstart, xend = datetime(2020,4,1),datetime(2021,6,1)
# xstart,xend = datetime(2018,5,1),datetime(2022,6,1)
xstart_winter, xend_winter = datetime(2020,10,1), datetime(2021,4,30)

outline_path   = "Greenland_data/russel/russel_domain.gpkg"
vel_dir = "Greenland_data/velocity/monthly/"

# plotting parameters
plt.rcParams['font.size'] = 21
annotate_fs = 24
lw = 2.5

# load model input independent of run
timeseries_path = f"parameter_runs/run_477/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
coords = smesh_.coordinates.dat.data  # mesh coordinates
B, H, S = load_topography(mesh_, sig=5)

fig = plt.figure(figsize=(20, 10))
gs  = fig.add_gridspec(2, 3, left=0.07, right=0.95, hspace=0.3, top=0.92, bottom=0.1, width_ratios=[1,0.01,0.6])

# gs_top = gs[0].subgridspec(1, 5, hspace=0.0, wspace=0.04, )
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,2])

for (i_r,(run_index,model_label,col)) in enumerate(zip(run_indices,model_labels,cols)):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, n_idx, _ = load_model_output(timeseries_path)
    dates_model, i_model = get_model_dates(n_idx, xstart, xend)

    print(run_index)

    # x0 = -174630.0
    # x, y = df.SpatialCoordinate(mesh_)
    # chi = df.conditional(x < x0, 1, 0)



    # extract q for the specified timestep
    qs    = []
    Qs    = []
    ms    = []
    budget = 0
    budgets = []
    sheet_volumes = []
    channel_volumes = []
    V   = df.FunctionSpace(mesh_, "CG", 1)
    V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
    V_sub = df.FunctionSpace(smesh_, "DG", 0)
    for (i_time, i_m) in enumerate(i_model):
        # input
        m     = df.Function(V)
        m.dat.data[:] = m_raw[:, i_m]
        m_tot = df.assemble(m * df.dx) # m^3 / yr
        m_in  = m_tot / 365 * 2              # m^3 (total over two days)
        ms.append(m_in / (365*24*3600))
        # output sheet
        q_vec = df.Function(V_vec)
        q_vec_x = q_raw[::2, i_m]
        q_vec_y = q_raw[1::2, i_m]
        q_vec.dat.data[:,0] = q_vec_x
        q_vec.dat.data[:,1] = q_vec_y
        # across_q = df.assemble((df.div(q_vec)) * df.dx) # m^3 / yr
        n = df.FacetNormal(mesh_)
        across_q = df.assemble(df.dot(q_vec, n) * df.ds(1))
        q_out = across_q / 365 * 2  # multiply by two days to get the total q in m^3 over that time period
        qs.append(q_out)
        h = df.Function(V)
        h.dat.data[:] = h_raw[:, i_m]
        sheet_volumes.append(df.assemble(h*df.dx)*1e-9)
        # output channels
        idx_outflow = get_outflow_index_skeleton_mesh(mesh_, smesh_)
        across_Q = np.sum(Q_raw[idx_outflow, i_m])
        Q_out = across_Q / 365 * 2
        Qs.append(Q_out)
        # channel size
        # S = df.Function(V_sub)
        # S.dat.data[:] = S_raw[:,i_m]
        # channel_volumes.append(df.assemble(S*df.dx(domain=smesh_)))

    qtot = (np.array(Qs)+np.array(qs)) / (365*24*3600)
    print(np.sum(np.array(ms)-qtot))
    # qtot = (np.array(qs)) / (365*24*3600)

    # if (run_index == 477) | (run_index == 312):
    # model_label = "alpha_s = "+ f"{hlp.get_params_from_input_file(run_index)["alpha_s"]}" + "h_r = "+ f"{hlp.get_params_from_input_file(run_index)["h_r"]}"

    if i_r == 0:
        ax1.plot(dates_model, ms, color="black", ls="dotted", label="Meltwater input")
        ax1.fill_between(dates_model, ms, qtot, color=col, facecolor=col, alpha=0.2)
    else:
        # ax1.fill_between(dates_model, ms, qtot, alpha=0.3, hatch="XX", facecolor="none")
        ax1.fill_between(dates_model, ms, qtot, edgecolor=col, alpha=0.3, hatch="XX", facecolor="none")
        # ax1.plot(dates_model, ms, color="black", ls="dotted", label="Meltwater input")
    ax1.plot(dates_model, qtot, label=model_label+" output", color=col, lw=lw)
    ax1.legend()
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    # ax1.annotate(f"{np.round(budget_neg*1e-8,decimals=2)}e8", xy=(0.6, 0.4+0.1*i_r), xycoords="axes fraction", color=col, fontsize=annotate_fs)
    ax1.set_ylabel("Total discharge "+ r"$(\mathrm{m^3\,a^{-1}})$")
    # ax1.set_yscale("log")
    # ax1.set_title("Domain-integrated water budget")
    panel_letter_annotation(ax1, 0, annotate_fs, xyc=(0.03, 1.04))

    if i_r == 1:
        model_label = "Reduced \nsheet flow"
    ax2.plot(dates_model, sheet_volumes, label=model_label, lw=lw, color=col)
    # ax2.plot(dates_model, channel_volumes, ls="dashed")
    # ax2.set_yscale('log')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax2.set_xlabel("Month of 2020 / 2021")
    ax2.set_ylabel("Sheet volume "+ r"$V_s$ $(\mathrm{km^3})$")
    if i_r == 1:
        # ax2.fill_betweenx([ax2.get_ylim()[0], ax2.get_ylim()[1]], xstart_winter, xend_winter, color="seagreen", alpha=0.18)
        ax2.vlines(xend_winter, -0.2, 2.3, color="black", ls="-.")
        ax2.annotate(r"$V_s^\mathrm{\,end}$", xy=[datetime(2021,5,3), 0.8]) #, xycoords="axes fraction")
    ax2.legend()
    panel_letter_annotation(ax2, 2, annotate_fs, xyc=(0.03, 1.04))


# xstart_summer, xstart_winter = datetime(2019,4,1), datetime(2019,11,1)



dUs_A = []
dUs_B = []
neg_budgets = []
m_basal = []
alpha_s = []
h_r     = []
V_sheet_max = []
V_sheet_sum = []
for (i_r,run_index) in enumerate(run_indices_suit):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    try:
        us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, n_idx, _ = load_model_output(timeseries_path)
        dates_model, i_model = get_model_dates(n_idx, xstart, xend)
        if len(dates_model)==0:
            continue
    except:
        continue

    print(run_index)

    float_params = hlp.get_params_from_input_file(run_index)
    m_b = float_params["m_basal"]

    # extract q for the specified timestep
    qs    = []
    Qs    = []
    ms    = []
    budget = 0
    budgets = []
    sheet_volumes = []
    V   = df.FunctionSpace(mesh_, "CG", 1)
    V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
    V_DG0 = df.FunctionSpace(smesh_, "DG", 0)
    for (i_time, i_m) in enumerate(i_model):
        # input
        m     = df.Function(V)
        m.dat.data[:] = m_raw[:, i_m] #- m_b
        m_tot = df.assemble(m * df.dx) # m^3 / yr
        m_in  = m_tot / 365 * 2              # m^3 (total over two days)
        ms.append(m_in)
        # output sheet
        q_vec = df.Function(V_vec)
        q_vec_x = q_raw[::2, i_m]
        q_vec_y = q_raw[1::2, i_m]
        q_vec.dat.data[:,0] = q_vec_x
        q_vec.dat.data[:,1] = q_vec_y
        across_q = df.assemble(df.div(q_vec) * df.dx) # m^3 / yr
        q_out = across_q / 365 * 2  # multiply by two days to get the total q in m^3 over that time period
        qs.append(q_out)
        # sheet volume
        h = df.Function(V)
        h.dat.data[:] = h_raw[:, i_m]
        sheet_volumes.append(df.assemble(h*df.dx))
        # output channels
        idx_outflow = get_outflow_index_skeleton_mesh(mesh_, smesh_)
        across_Q = np.sum(Q_raw[idx_outflow, i_m])
        Q_out = across_Q / 365 * 2
        Qs.append(Q_out)
        # total budget
        budget = m_in - q_out - Q_out
        budgets.append(budget)

    budget_neg = np.sum(np.array(budgets)[np.where(np.array(budgets) > 0)])
    budget_q = np.sum(np.array(Qs)[np.where(np.array(budgets) < 0)])
    # qtot = np.array(Qs)+np.array(qs)

    # parameters
    m_basal.append(m_b)
    alpha_s.append(float_params["alpha_s"])
    h_r.append(float_params["h_r"])
    # get winter speedup in specific locatiodUsn
    U_monthly = get_monthly_means_model(mesh_, run_index, xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]])
    dU = fit_linear_slope(U_monthly)
    dUs_A.append(dU)
    U_monthly = get_monthly_means_model(mesh_, run_index, xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]])
    dU = fit_linear_slope(U_monthly)
    dUs_B.append(dU)
    neg_budgets.append(budget_q)
    i_hmax = np.argmax(np.array(sheet_volumes))
    V_sheet_sum.append(sheet_volumes[-1]*1e-9) #/sheet_volumes[i_hmax])
    # V_sheet_sum.append(np.sum(np.array(sheet_volumes)))
    #
    xc_input, yc_input = prepare_location_inputs(xc, yc)
    h_winter_loc = load_model_timeseries(mesh_, h_raw, xc_input, yc_input, i_model)[:, 0]
    V_sheet_max.append(h_winter_loc.mean())


markersz = 20

im = ax3.scatter(V_sheet_sum, dUs_A, markersz, color="black", marker="x")
ax3.set_ylabel(r"$\Delta u_\mathrm{A}$ " + r"($\mathrm{m\,a^{-1}}$)")
panel_letter_annotation(ax3, 1, annotate_fs, xyc=(0.03, 1.04))
ax3.set_xscale('log')
ax3.set_title("Point A")
# fig.colorbar(im, ax=ax3)

im = ax4.scatter(V_sheet_sum, dUs_B, markersz, color="black", marker="x")
# ax4.set_xscale('log')
ax4.set_xlabel(r"$V_s^\mathrm{\,end}$ ($\mathrm{km^3}$)")
ax4.set_ylabel(r"$\Delta u_\mathrm{B}$ " + r"($\mathrm{m\,a^{-1}}$)")
panel_letter_annotation(ax4, 3, annotate_fs, xyc=(0.03, 1.04))
ax4.set_xscale('log')
ax4.set_title("Point B")
# fig.colorbar(im, ax=ax4)

plt.savefig(f"plotting/main_figures/f05.jpg")
