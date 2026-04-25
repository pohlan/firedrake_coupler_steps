import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

run_index = 126
timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
flowlines_path = "Greenland_data/russel/flowlines.gpkg"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"

profile_width = 1e3

ds = 2.5e3
# gl = 5
glacier_ids = [1,3,4]
n_glaciers = len(glacier_ids)
# ss = [10e3,30e3] #,43e3]  # km away from terminus at which to plot the time series
d_upglacier = [15e3,8e3,15e3,20e3]

xstart,xend = datetime(2018,1,2),datetime(2022,12,30)

# load model file meshes
mesh_, smesh_ = get_meshes(timeseries_path)

# load model output from hdf5 file
us_raw, m_raw, phi_raw, q_raw, Q_raw, n_idx = load_model_output(timeseries_path)

# load topography
B, H, S = load_topography(mesh_)

# load velocity observations
files   = os.listdir(vel_dir)
sorted_dates, U_obs, U_mask = load_vel_obs(vel_dir, files, mesh_)

# set colormap for timeseries, color=distance along profile
# colors = plt.cm.Purples(np.linspace(0, 1, 5))[1:]
# colors = plt.cm.tab20b.colors[12:]
colors = ["coral","cornflowerblue","yellowgreen","palevioletred"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
# Increase font sizes
plt.rcParams['font.size'] = 18
lw = 2.5

fig = plt.figure(figsize=(len(glacier_ids)*8,15))
# Create two gridspecs: one for rows 0-2 (tight), one for row 3 (with gap)
# Each row should have equal height; gap between row 3 and 4 is 0.05
# Row height = (0.95 - 0.05 - 0.05) / 4 = 0.2125
gs_top = fig.add_gridspec(3, len(glacier_ids), left=0.1, right=0.95, top=0.95, bottom=0.3125, hspace=0.0, wspace=0.1)
gs_bottom = fig.add_gridspec(1, len(glacier_ids), left=0.1, right=0.95, top=0.2625, bottom=0.05, hspace=0.0, wspace=0.1)
n_panels = len(glacier_ids)*4

# Store ax2 objects from first row to share y-axis
ax2_list = []

for (ig,(gl,splus,color)) in enumerate(zip(glacier_ids, d_upglacier, colors)):
    # flow linestrings don't have the same numbering as gl; also gl starts at 1
    fl = [0,4,1,2,3][gl-1]

    # load flowlines and compute along flowline functions s
    s, s_sub = get_s_functions(flowlines_path, fl, mesh_, smesh_, profile_width)

    # time series, at certain slice of flowline
    q, Q, Us, pw_pi, m = get_timestamps(mesh_, smesh_, B, H, us_raw, phi_raw, q_raw, Q_raw, m_raw, n_idx)

    # get model time series
    start_date = datetime(2018, 1, 1)
    dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
    i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]

    # plot velocity time series
    ax1 = fig.add_subplot(gs_top[0, ig])
    ax2 = plot_vel_timeseries(mesh_, splus, s, dates_model, Us, m, sorted_dates, U_obs, U_mask, xstart, xend, i_model, color, lw, ds, ig, n_glaciers)
    ax2_list.append(ax2)
    if ig > 0:
        ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax1.set_xticks([])
    # if ig > 0:


    ax2 = fig.add_subplot(gs_top[1, ig])
    ymin, ymax = 0, 1
    # for splus in ss:
    pw_pi_time = []
    for pw_pi_i in pw_pi:
        pw_pi_time.append(get_variable(pw_pi_i, mesh_, s, [splus-ds/2,splus+ds/2])[0])
    plt.plot(dates_model, pw_pi_time, label="", lw=lw, color=color)
    ymin = min(np.min(np.array(pw_pi_time)[i_model]),ymin)
    ymax = max(np.max(np.array(pw_pi_time)[i_model]),ymax)
    ax = plt.gca()
    format_ax(ax, xstart, xend, ig, ylims=(-1.02,1.1), ylabel="Pw / Pi" if ig == 0 else "", draw_legend=False)
    ax.set_xlabel("")
    ax.set_xticks([])
    # if ig > 0:
    #     ax.tick_params(axis='y', labelleft=False)

    ax3 = fig.add_subplot(gs_top[2, ig])
    # for (ic,splus) in enumerate(ss):
    qi_time = []
    Qi_time = []
    for (Qi,qi) in zip(Q,q):
        qi_time.append(get_variable(qi, mesh_, s, [splus-ds/2,splus+ds/2])[0])
        Qi_time.append(get_Q(Qi, smesh_, s_sub, [splus-ds/2,splus+ds/2])[0])
    qi_time = np.array(qi_time) # np.array(qi_time)/(1.793e-6*365*24*3600)  # Re
    Qi_time = np.array(Qi_time)
    plt.plot(dates_model, Qi_time, label=f"Q (channels)", lw=lw, ls="-.", color=color)
    plt.plot(dates_model, qi_time, label=f"q (sheets)", lw=lw, color=color)
    ymin = min(np.min(qi_time[i_model]),np.min(Qi_time[i_model]))
    ymax = max(np.max(qi_time[i_model]),np.max(Qi_time[i_model]))
    plt.yscale("log")
    ax = plt.gca()
    format_ax(ax, xstart, xend, ig, ylims=(0.5,5e9), ylabel="Discharge (m^3/s)" if ig == 0 else "")
    # ax.set_xlabel("Date" if ig == 0 else "")
    # if ig > 0:
    #     # ax.set_yticks([])
    #     ax.tick_params(axis='y', labelleft=False)

    ax4 = fig.add_subplot(gs_bottom[0, ig])
    # distance along flowlines
    s0s = np.arange(0,np.max(s.dat.data_ro),step=ds)
    s0s_avg = (s0s[:-1]+s0s[1:]) / 2
    d_along = (s0s_avg - s0s_avg[0] ) / 1e3
    print(f"# bins along profile: {len(s0s)}")
    Si = get_variable(S, mesh_, s, s0s)
    Bi = get_variable(B, mesh_, s, s0s)
    plt.plot(d_along, Si, label="Surface", color="grey", lw=lw)
    plt.plot(d_along, Bi, label="Bed", color="Black", lw=lw)
    plt.vlines(splus/1e3,-500,max(Si)*1.5,ls="dashed",color=color, lw=lw)
    plt.ylim(-max(Si)*0.05,max(Si)*1.05)
    plt.xlim(-5,50)
    plt.xlabel("Distance along profile (km)")
    if ig > 0:
        ax4.set_ylabel("")
    plt.ylabel("Elevation (m)" if ig == 0 else "")
    if ig > 0:
        # ax4.set_yticks([])
        ax4.tick_params(axis='y', labelleft=False, length=7, width=2)
    plt.legend(loc="center right")

# Share y-axis for all ax2 in first row
if ax2_list:
    # Get the global min/max of all ax2 y-limits
    all_ylims = [ax.get_ylim() for ax in ax2_list]
    global_ymin = min([ylim[0] for ylim in all_ylims])
    global_ymax = max([ylim[1] for ylim in all_ylims])
    for ax in ax2_list:
        ax.set_ylim(global_ymin, global_ymax)

plt.tight_layout()
plt.savefig(f"plotting/output/timeseries_run{run_index}_d{int(d_upglacier[0]/1e3)}.jpg", dpi=150, bbox_inches='tight')
