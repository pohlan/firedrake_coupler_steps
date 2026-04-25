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

profile_width = 3e3

ds = 2.5e3
# gl = 5
glacier_ids = [1,2,3,5]
# ss = [10e3,30e3] #,43e3]  # km away from terminus at which to plot the time series
d_upglacier = [10e3,10e3,10e3,10e3]

xstart,xend = datetime(2019,1,2),datetime(2021,12,30)

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

plt.figure(figsize=(len(glacier_ids)*6,15))
n_panels = len(glacier_ids)*4

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
    plt.subplot(4, len(glacier_ids), ig+1)
    plot_vel_timeseries(mesh_, splus, s, dates_model, Us, m, sorted_dates, U_obs, U_mask, xstart, xend, i_model, color, lw, ds)

    plt.subplot(4, len(glacier_ids), len(glacier_ids)+ig+1)
    ymin, ymax = 0, 1
    # for splus in ss:
    pw_pi_time = []
    for pw_pi_i in pw_pi:
        pw_pi_time.append(get_variable(pw_pi_i, mesh_, s, [splus-ds/2,splus+ds/2])[0])
    plt.plot(dates_model, pw_pi_time, label=f"{splus*1e-3:.0f} km", lw=lw, color=color)
    ymin = min(np.min(np.array(pw_pi_time)[i_model]),ymin)
    ymax = max(np.max(np.array(pw_pi_time)[i_model]),ymax)
    ax = plt.gca()
    format_ax(ax, xstart, xend, ylims=(ymin,ymax), ylabel="Pw / Pi")

    plt.subplot(4, len(glacier_ids), 2*len(glacier_ids)+ig+1)
        # for (ic,splus) in enumerate(ss):
    qi_time = []
    Qi_time = []
    for (Qi,qi) in zip(Q,q):
        qi_time.append(get_variable(qi, mesh_, s, [splus-ds/2,splus+ds/2])[0])
        Qi_time.append(get_Q(Qi, smesh_, s_sub, [splus-ds/2,splus+ds/2])[0])
    qi_time = np.array(qi_time) # np.array(qi_time)/(1.793e-6*365*24*3600)  # Re
    Qi_time = np.array(Qi_time)
    plt.plot(dates_model, qi_time, label=f"q (sheets)", lw=lw, color=color)
    plt.plot(dates_model, Qi_time, label=f"Q (channels)", lw=lw, ls="-.", color=color)
    ymin = min(np.min(qi_time[i_model]),np.min(Qi_time[i_model]))
    ymax = max(np.max(qi_time[i_model]),np.max(Qi_time[i_model]))
    plt.yscale("log")
    ax = plt.gca()
    format_ax(ax, xstart, xend, ylims=(ymin,ymax), ylabel="Discharge (m^3/s)")

    plt.subplot(4, len(glacier_ids), 3*len(glacier_ids)+ig+1)
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
    plt.ylabel("Elevation (m)")
    plt.legend(loc="center right")

plt.tight_layout()
plt.savefig(f"plotting/output/timeseries_run{run_index}_d{d_upglacier[0]}.jpg", dpi=150, bbox_inches='tight')
