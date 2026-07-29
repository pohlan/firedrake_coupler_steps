import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

flowlines_dir = "Greenland_data/profiles/"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"

# profile width, which runs and glaciers to plot
profile_width = 4.5e3
glacier_ids = [1,1,4,4]
n_glaciers = len(glacier_ids)
ds_upglacier = [10e3,30e3,15e3,23e3]
deltas_upglacier = [500,500,500,500]
glacier_names = ["Isunnguata Sermia", "Isunnguata Sermia", "Isorlersuup", "Isorlersuup"]
# glacier_names = ["Isunnguata Sermia", "Isunnguata Sermia", "Russel", "Russel"]
run_indices = [311,420]
model_start_years = [2014,2014] # ToDo: automate this
model_labels = ["baseline", "reduced \nsheet flow"]

# time period to plot
xstart,xend = datetime(2017,1,2),datetime(2022,12,30)

# load glacier flowline coordinates and distances along profile
x_points = []
y_points = []
for (gl_name, d, delta) in zip(glacier_names, ds_upglacier, deltas_upglacier):
    dists, xx, yy = get_profile(flowlines_dir, gl_name)
    # get indices closest to the points to plot
    idx_profiles = np.where(abs(dists - d)<delta)
    x_points.append(xx[idx_profiles])
    y_points.append(yy[idx_profiles])

# load model stuff not run-specific
timeseries_path = f"parameter_runs/run_{run_indices[0]}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
B, H, S = load_topography(mesh_)

# get velocity observation files sorted after date
files   = os.listdir(vel_dir)
obs_dates, obs_files = get_obs_files(files, xrange=(xstart,xend))
Uobs_glaciers = load_obs_timeseries(vel_dir, obs_files, x_points, y_points)

# plotting parameters
colors = ["coral","cornflowerblue","yellowgreen","palevioletred"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
plt.rcParams['font.size'] = 18
lw = 2.5
annotate_fs = 18

# prepare figure, gridspecs
fig = plt.figure(figsize=(len(glacier_ids)*7,13))
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

for (run_index,color,model_label,model_start) in zip(run_indices, colors, model_labels, model_start_years):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    # load model output from hdf5 file
    mesh_, smesh_ = get_meshes(timeseries_path)
    B, H, S = load_topography(mesh_)
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, n_idx, _ = load_model_output(timeseries_path)

    for (ig,(gl, gl_name, d_along_profile, delta, Uobs, xc, yc)) in enumerate(zip(glacier_ids, glacier_names, ds_upglacier, deltas_upglacier, zip(*Uobs_glaciers), x_points, y_points)):
        # flow linestrings don't have the same numbering as gl; also gl starts at 1
        fl = [0,4,1,2,3][gl-1]

        # get model time vector
        start_date = datetime(model_start, 1, 1)
        dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
        i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]
        dates_model = np.array(dates_model)[i_model]
        dates_model_pd = pd.to_datetime(dates_model)

        # plot velocity time series
        ax1 = axes[0,ig]
        U_model = load_model_timeseries(mesh_, us_raw, [xc], [yc], i_model)
        m_time  = load_model_timeseries(mesh_, m_raw, [xc], [yc], i_model)[:,0]
        plot_vel_timeseries(dates_model, U_model, m_time, obs_dates, Uobs, xstart, xend, color, lw, ig, n_glaciers, model_label, ax1)
        if ig > 0:
            ax1.set_ylabel("")
        ax1.set_xlabel("")
        ax1.set_xticks([])

        # annotations for distance along profile
        ax1.annotate(f"{int(d_along_profile*1e-3)} km", xy=[0.5,1.1], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")

        # plot Pw / Pi time series
        ax2 = axes[1, ig]
        Pw_Pi = np.mean(load_model_timeseries_Pw_Pi(mesh_, phi_raw, xc, yc, i_model, B, H), axis=1)
        # convert to dataframe in order to calculate 10-day mean
        df_PwPi = pd.DataFrame(Pw_Pi, index=dates_model_pd)
        Pw_Pi_resampled = df_PwPi.resample('10D').mean()
        ax2.plot(Pw_Pi_resampled.index, Pw_Pi_resampled.values, label=model_label, lw=lw, color=color)
        format_ax(ax2, xstart, xend, ig, panel_idx=4+ig, ylims=(-0.41,1.1), ylabel=r"$p_w / p_i$" if ig == 0 else "")
        ax2.set_xlabel("")
        ax2.set_xticks([])
        # legend outside of plots, in separate axis
        if (run_index == run_indices[-1]) & (ig==n_glaciers-1):
            lax = fig.add_subplot(gs[3, 5])
            lines = ax1.get_lines()  # ax1 so that observations are in there too
            labels = [h.get_label() for h in lines]
            lax.legend(handles=lines, labels=labels, loc=(-0.05,0.2), frameon=False)
            lax.axis("off")
        if (run_index == run_indices[-1]) & (ig==0):
            ax2.annotate("January", xy=[datetime(2020,2,1),-0.31], rotation=90)

        # plot discharges q and Q
        ax3 = axes[2, ig]
        #  compute along flowline functions s
        dists, xx, yy = get_profile(flowlines_dir, gl_name)
        s, s_sub = get_s_functions(dists, xx, yy, mesh_, smesh_, profile_width)
        # get discharge timeseries, integrated over profile_width * delta
        q, Q = load_model_timeseries_discharges(mesh_, smesh_, q_raw, Q_raw, i_model, s, s_sub, d_along_profile, delta)

        s_per_yr = 3600*24*365
        # convert to dataframe in order to calculate 10-day mean
        df_q = pd.DataFrame(q, index=dates_model_pd)
        q_resampled = df_q.resample('10D').mean()
        df_Q = pd.DataFrame(Q, index=dates_model_pd)
        Q_resampled = df_Q.resample('10D').mean()

        ax3.plot(Q_resampled.index, Q_resampled.values/s_per_yr, lw=lw, ls="-.", color=color)
        ax3.plot(q_resampled.index, q_resampled.values/s_per_yr, lw=lw, color=color)
        ymin = min(np.min(q),np.min(Q))
        ymax = max(np.max(q),np.max(Q))
        ax3.set_yscale("log")
        format_ax(ax3, xstart, xend, ig, panel_idx=2*4+ig, ylims=(5e-7,300), ylabel=r"Discharge ($\mathrm{m^3\,s^{-1}}$)" if ig == 0 else "", draw_legend=False)
        if (run_index == run_indices[-1]) & (ig==n_glaciers-1):
            lax = fig.add_subplot(gs[4, 5])
            lax.plot([], [], color='grey', linestyle="-.", label="Q (channels)")
            lax.plot([], [], color='grey', linestyle="solid", label="q (sheets)")
            lax.legend(loc=(-0.05,0.4), title="Discharge", frameon=False)
            lax.axis("off")

# Share y-axis for all velocity plots (first row)
all_ylims = [ax.get_ylim() for ax in axes[0,:]]
global_ymin = min([ylim[0] for ylim in all_ylims])
global_ymax = max([ylim[1] for ylim in all_ylims])
for ax in axes[0,:]:
    ax.set_ylim(global_ymin, global_ymax)

plt.tight_layout(pad=0.4)
plt.savefig(f"plotting/output/timeseries_run{run_indices[0]}_{run_indices[1]}_d{int(ds_upglacier[0]*1e-3)}_{int(ds_upglacier[1]*1e-3)}.jpg", dpi=150, bbox_inches='tight')
# plt.savefig(f"plotting/main_figures/f03.jpg")
