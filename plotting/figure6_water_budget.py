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

run_indices = [312,420]
model_labels  = ["Baseline", "Reduced \nsheet flow", "any", "any", "any"]
colors = ["orange", "blue", "any", "any", "any"]

xstart,xend = datetime(2018,5,1),datetime(2022,6,1)
xstart_winter,xend_winter = datetime(2019,11,1),datetime(2020,4,1)

outline_path   = "Greenland_data/russel/russel_domain.gpkg"
vel_dir = "Greenland_data/velocity/monthly/"

# plotting parameters
plt.rcParams['font.size'] = 15
annotate_fs = 15

# load model input independent of run
timeseries_path = f"parameter_runs/run_{run_indices[0]}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
coords = smesh_.coordinates.dat.data  # mesh coordinates
B, H, S = load_topography(mesh_, sig=5)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,5))

dUs = []
neg_budgets = []
for (i_r,(run_index,model_label,col)) in enumerate(zip(run_indices,model_labels,colors)):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, n_idx, _ = load_model_output(timeseries_path)

    dates_model, i_model = get_model_dates(n_idx, xstart, xend)


    x0 = -174630.0
    x, y = df.SpatialCoordinate(mesh_)
    chi = df.conditional(x < x0, 1, 0)

    # extract q for the specified timestep
    qs    = []
    Qs    = []
    ms    = []
    budget = 0
    budgets = []
    V   = df.FunctionSpace(mesh_, "CG", 1)
    V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
    V_DG0 = df.FunctionSpace(smesh_, "DG", 0)
    for (i_time, i_m) in enumerate(i_model):
        # input
        m     = df.Function(V)
        m.dat.data[:] = m_raw[:, i_m]
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
        # output channels
        idx_outflow = get_outflow_index_skeleton_mesh(mesh_, smesh_)
        across_Q = np.sum(Q_raw[idx_outflow, i_m])
        Q_out = across_Q / 365 * 2
        Qs.append(Q_out)
        # total budget
        budget = m_in - q_out - Q_out
        budgets.append(budget)

    qtot = np.array(Qs)+np.array(qs)

    if (run_index == 420) | (run_index == 312):
        ax1.plot(dates_model, qtot, color=col, label=model_label)
        if i_r == 0:
            ax1.fill_between(dates_model, ms, qtot, color=col, alpha=0.3)
        else:
            ax1.fill_between(dates_model, ms, qtot, edgecolor=col, alpha=0.3, hatch="XX", facecolor="none")
        budget_neg = np.sum(np.array(budgets)[np.where(np.array(budgets) < 0)])
        ax1.annotate(f"{np.round(budget_neg*1e-8,decimals=2)}e8", xy=(0.6, 0.4+0.1*i_r), xycoords="axes fraction", color=col, fontsize=annotate_fs)

    # get winter speedup in specific location
    dU = get_mean_winter_speedup(mesh_, run_index, xstart_winter, xend_winter, xc, yc)
    dUs.append(dU)
    neg_budgets.append(budget_neg)

ax1.plot(dates_model, ms, color="black", ls="dotted", label="meltwater input")
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax1.legend()


ax2.scatter(neg_budgets, dUs)

fig.savefig("test2.jpg")
