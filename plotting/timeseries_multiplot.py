import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# beta2     = 5.0
# timeseries_path = f"test_inversion/forward_beta2_{beta2}e5/time_series.h5"
flowlines_path = "Greenland_data/russel/flowlines.gpkg"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"

# profile width, which runs and glaciers to plot
profile_width = 2e3
ds = 2.5e3
glacier_ids = [1,1,4,4]
n_glaciers = len(glacier_ids)
d_upglacier = [10e3,25e3,10e3,25e3]
glacier_names = ["Isunnguata Sermia", "", "Isorlersuup", ""]
run_indices = [177,178]
model_labels = ["M0", "M1"]

# time period to plot
xstart,xend = datetime(2018,1,2),datetime(2021,12,30)

# load model stuff not run-specific
timeseries_path = f"parameter_runs/run_{run_indices[0]}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
B, H, S = load_topography(mesh_)

# load velocity observations
files   = os.listdir(vel_dir)
sorted_dates, U_obs, U_mask, _ = load_vel_obs(vel_dir, files, mesh_)

# plotting parameters
colors = ["coral","cornflowerblue","yellowgreen","palevioletred"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
plt.rcParams['font.size'] = 18
lw = 2.5
annotate_fs = 18

# prepare figure, gridspecs
fig = plt.figure(figsize=(len(glacier_ids)*7,15))
gs = fig.add_gridspec(5, len(glacier_ids)+2, left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.0, wspace=0.1, width_ratios=[6, 6, 0.2, 6, 6, 1], height_ratios=[2,2,10,10,10])
axes = np.array([[fig.add_subplot(gs[j, i]) for i in [0,1,3,4]] for j in range(2,5)])

# set title for the two glaciers, group the columns
for (i,glacier_name) in enumerate([glacier_names[0],glacier_names[2]]):
    i0 = i*3
    topax = fig.add_subplot(gs[0, i0:i0+2])
    topax.fill_between([0,1],0,1, color="grey", alpha=0.1)
    topax.set_xlim(0,1)
    topax.axis("off")
    topax.annotate(glacier_name, xy=[0.5,0.5], fontsize=annotate_fs*1.3, fontweight="bold", ha="center", va="center")

for (run_index,color,model_label) in zip(run_indices,colors,model_labels):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    # load model output from hdf5 file
    us_raw, m_raw, phi_raw, q_raw, Q_raw, n_idx = load_model_output(timeseries_path)

    for (ig,(gl,splus,glacier_name)) in enumerate(zip(glacier_ids, d_upglacier, glacier_names)):
        # flow linestrings don't have the same numbering as gl; also gl starts at 1
        fl = [0,4,1,2,3][gl-1]

        # load flowlines and compute along flowline functions s
        s, s_sub = get_s_functions(flowlines_path, fl, mesh_, smesh_, profile_width)

        # time series, at certain slice of flowline
        q, Q, Us, pw_pi, m = get_timestamps(mesh_, smesh_, B, H, us_raw, phi_raw, q_raw, Q_raw, m_raw, n_idx)

        # get model time stamps
        start_date = datetime(2016, 1, 1)
        dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
        i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]

        # plot velocity time series
        ax1 = axes[0,ig]
        plot_vel_timeseries(mesh_, splus, s, dates_model, Us, m, sorted_dates, U_obs, U_mask, xstart, xend, i_model, color, lw, ds, ig, n_glaciers, model_label, ax1)
        if ig > 0:
            ax1.set_ylabel("")
        ax1.set_xlabel("")
        ax1.set_xticks([])

        # annotations for distance along profile
        ax1.annotate(f"{int(splus*1e-3)} km", xy=[0.5,1.1], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")

        ax2 = axes[1, ig]
        pw_pi_time = []
        for pw_pi_i in pw_pi:
            pw_pi_time.append(get_variable(pw_pi_i, mesh_, s, [splus-ds/2,splus+ds/2])[0])
        ax2.plot(dates_model, pw_pi_time, label=model_label, lw=lw, color=color)
        format_ax(ax2, xstart, xend, ig, panel_idx=4+ig, ylims=(-0.4,1.1), ylabel=r"$p_w / p_i$" if ig == 0 else "")
        ax2.set_xlabel("")
        ax2.set_xticks([])
        # legend outside of plots, in separate axis
        if (run_index == run_indices[-1]) & (ig==n_glaciers-1):
            lax = fig.add_subplot(gs[3, 5])
            lines = ax1.get_lines()  # ax1 so that observations are in there too
            labels = [h.get_label() for h in lines]
            lax.legend(handles=lines, labels=labels, loc=(-0.05,0.4), frameon=False)
            lax.axis("off")

        ax3 = axes[2, ig]
        qi_time = []
        Qi_time = []
        for (Qi,qi) in zip(Q,q):
            qi_time.append(get_q(qi, mesh_, s, [splus-ds/2,splus+ds/2])[0])
            Qi_time.append(get_Q(Qi, smesh_, s_sub, [splus-ds/2,splus+ds/2])[0])
        qi_time = np.array(qi_time) # np.array(qi_time)/(1.793e-6*365*24*3600)  # Re
        Qi_time = np.array(Qi_time)
        s_per_yr = 3600*24*365
        ax3.plot(dates_model, Qi_time/s_per_yr, lw=lw, ls="-.", color=color)
        ax3.plot(dates_model, qi_time/s_per_yr, lw=lw, color=color)
        ymin = min(np.min(qi_time[i_model]),np.min(Qi_time[i_model]))
        ymax = max(np.max(qi_time[i_model]),np.max(Qi_time[i_model]))
        ax3.set_yscale("log")
        format_ax(ax3, xstart, xend, ig, panel_idx=2*4+ig, ylims=(1e-8,50), ylabel=r"Discharge ($\mathrm{m^3\,s^{-1}}$)" if ig == 0 else "", draw_legend=False)
        if (run_index == run_indices[-1]) & (ig==n_glaciers-1):
            lax = fig.add_subplot(gs[4, 5])
            lax.plot([], [], color='grey', linestyle="-.", label="Q (channels)")
            lax.plot([], [], color='grey', linestyle="solid", label="q (sheets)")
            lax.legend(loc=(-0.05,0.6), title="Discharge", frameon=False)
            lax.axis("off")
        if (run_index == run_indices[-1]) & (ig==0):
            ax3.annotate("January", xy=[datetime(2018,10,25),3e-8], rotation=90)

# Share y-axis for all velocity plots (first row)
all_ylims = [ax.get_ylim() for ax in axes[0,:]]
global_ymin = min([ylim[0] for ylim in all_ylims])
global_ymax = max([ylim[1] for ylim in all_ylims])
for ax in axes[0,:]:
    ax.set_ylim(global_ymin, global_ymax)

plt.tight_layout(pad=0.4)
plt.savefig(f"plotting/output/timeseries_run{run_indices[0]}_{run_indices[1]}.jpg", dpi=150, bbox_inches='tight')
# plt.savefig(f"plotting/main_figures/f03.jpg")
