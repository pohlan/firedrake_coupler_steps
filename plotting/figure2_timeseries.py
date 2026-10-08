import os
import sys
import glob
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

flowlines_dir = "Greenland_data/profiles/"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"

# whether to plot Fig. 2 in main document or same figure for different points as in supplements
fig_mode = ["main","supp"][0]

# profile width, which runs and glaciers to plot
profile_width = 4.5e3
deltas_upglacier = [500,500,500,500]
wspace_size = 0.05
if fig_mode == "main":
    ds_upglacier = [10e3,35e3,10e3,16e3]
    glacier_names = ["Isunnguata Sermia", "Isunnguata Sermia", "Russel", "Isorlersuup"]
    width_ratios = [6, 6, wspace_size, 6, wspace_size, 6]
    title_bar_lengths = [2,1,1]
    pw_lims = (-0.3,0.98)
    y_Jan_annot = -0.25
    speed_lims = (-60,130)
    figname = "f02"
else:
    ds_upglacier = [10e3,30e3,10e3,30e3]
    glacier_names = ["Ørkendalen", "Ørkendalen", "Glacier 5", "Glacier 5"]
    width_ratios = [6, 6, wspace_size, 6, 0.0, 6, 1]
    title_bar_lengths = [2,3]
    pw_lims = (-1.2,0.98)
    y_Jan_annot = 0.4
    speed_lims = (-120,140)
    figname = "fS02"

n_glaciers = len(glacier_names)
run_indices = [314,478]
model_labels = ["Baseline", "Reduced \nsheet flow"]
model_start_years = [2014,2014]
# model_labels = ["Larger domain", "Reduced \nsheet flow"]
# model_labels = ["Reduced \nsheet flow", "With moulins"]
# model_labels = ["Reduced \nsheet flow", r"Locally increased \n$k_s$"]

# time period to plot
xstart,xend = datetime(2019,1,2),datetime(2022,12,30)
# xstart,xend = datetime(2015,1,2),datetime(2019,12,30)

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
files   = glob.glob(vel_dir+"*vv*.tif")
obs_dates, obs_files = get_obs_files(files, xrange=(xstart,xend))
Uobs_glaciers = load_obs_timeseries(obs_files, x_points, y_points)

# plotting parameters
colors = ["coral","cornflowerblue","yellowgreen","palevioletred"]
# colors = ["peru","cornflowerblue"]        # smaller domain
# colors = ["indianred","cornflowerblue"]   # larger domain
# colors = ["cornflowerblue","yellowgreen"] # moulin run
# colors = ["cornflowerblue","palevioletred"] # locally increased k_s
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
plt.rcParams['font.size'] = 30
lw = 3.5
annotate_fs = 30

# prepare figure, gridspecs
fig = plt.figure(figsize=(n_glaciers*7,16))
gs = fig.add_gridspec(6, n_glaciers+2, left=0.065, right=0.955, top=0.99, bottom=0.0, hspace=0.0, wspace=0.07, width_ratios=width_ratios, height_ratios=[2,2,10,7,7,6])
axes = np.array([[fig.add_subplot(gs[j, i]) for i in [0,1,3,5]] for j in range(2,5)])

# set title for the two glaciers, group the columns
i0=0
unique_gl_names = np.array(glacier_names)[np.sort(np.unique(glacier_names, return_index=True)[1])]  # np.unique() itself changes order, need stuff around to preserve it
for (i,(glacier_name,l_title)) in enumerate(zip(unique_gl_names,title_bar_lengths)):
    topax = fig.add_subplot(gs[0, i0:i0+l_title])
    topax.fill_between([0,1],0,1, color="grey", alpha=0.1)
    topax.set_xlim(0,1)
    topax.axis("off")
    topax.annotate(glacier_name, xy=[0.5,0.5], fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    i0 += l_title+1

for (run_index,color,model_label,model_start) in zip(run_indices, colors, model_labels, model_start_years):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    # load model output from hdf5 file
    mesh_, smesh_ = get_meshes(timeseries_path)
    B, H, S = load_topography(mesh_)
    us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, S_raw, n_idx, _ = load_model_output(timeseries_path)

    for (ig,(gl_name, d_along_profile, delta, Uobs, xc, yc)) in enumerate(zip(glacier_names, ds_upglacier, deltas_upglacier, zip(*Uobs_glaciers), x_points, y_points)):
        # get model time vector
        start_date = datetime(model_start, 1, 1)
        dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
        i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]
        dates_model = np.array(dates_model)[i_model]
        # dates_model_pd = pd.to_datetime(dates_model)

        # plot velocity time series
        ax1 = axes[0,ig]
        U_model = load_model_timeseries(mesh_, us_raw, [xc], [yc], i_model)
        m_time  = load_model_timeseries(mesh_, m_raw, [xc], [yc], i_model)[:,0]
        plot_vel_timeseries(dates_model, U_model, m_time, obs_dates, Uobs, xstart, xend, color, lw, ig, n_glaciers, model_label, ax1, speed_lims, fs=annotate_fs*0.9)
        if ig > 0:
            ax1.set_ylabel("")
        ax1.set_xlabel("")
        ax1.set_xticks([])

        # annotations for distance along profile
        ax1.annotate(f"{int(d_along_profile*1e-3)} km", xy=[0.5,1.1], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs*0.9, fontweight="bold", ha="center", va="center")

        # plot Pw / Pi time series
        ax2 = axes[1, ig]
        Pw_Pi = np.mean(load_model_timeseries_Pw_Pi(mesh_, phi_raw, xc, yc, i_model, B, H), axis=1)
        # convert to dataframe in order to calculate 10-day mean
        # df_PwPi = pd.DataFrame(Pw_Pi, index=dates_model_pd)
        # Pw_Pi_resampled = df_PwPi.resample('10D').mean()
        ax2.plot(dates_model, Pw_Pi, label=model_label, lw=lw, color=color)
        format_ax(ax2, xstart, xend, ig, panel_idx=4+ig, ylims=pw_lims, ylabel=r"$p_w / p_i$" if ig == 0 else "", fontsize=annotate_fs*0.9, y_panellabel=6/7)
        ax2.set_xlabel("")
        ax2.set_xticks([])
        # legend outside of plots, in separate axis
        if (run_index == run_indices[-1]) & (ig==n_glaciers-1):
            lax = fig.add_subplot(gs[5, 0])
            lines = ax1.get_lines()  # ax1 so that observations are in there too
            labels = [h.get_label() for h in lines]
            lax.legend(handles=lines, labels=labels, loc=(0.25,0.2), frameon=False, handlelength=1.5, ncols=3)
            lax.axis("off")
        if (run_index == run_indices[-1]) & (ig==0):
            ax2.annotate("January", xy=[datetime(2020,2,1),y_Jan_annot], rotation=90, fontsize=annotate_fs*0.9)

        # plot discharges q and Q
        ax3 = axes[2, ig]
        #  compute along flowline functions s
        dists, xx, yy = get_profile(flowlines_dir, gl_name)
        s, s_sub = get_s_functions(dists, xx, yy, mesh_, smesh_, profile_width)
        # get discharge timeseries, integrated over profile_width * delta
        q, Q = load_model_timeseries_discharges(mesh_, smesh_, q_raw, Q_raw, i_model, s, s_sub, d_along_profile, delta)

        s_per_yr = 3600*24*365
        # convert to dataframe in order to calculate 10-day mean
        # df_q = pd.DataFrame(q, index=dates_model_pd)
        # q_resampled = df_q.resample('10D').mean()
        # df_Q = pd.DataFrame(Q, index=dates_model_pd)
        # Q_resampled = df_Q.resample('10D').mean()

        ax3.plot(dates_model, Q/s_per_yr, lw=lw, ls="dashed", color=color)
        ax3.plot(dates_model, q/s_per_yr, lw=lw, color=color)
        ymin = min(np.min(q),np.min(Q))
        ymax = max(np.max(q),np.max(Q))
        ax3.set_yscale("log")
        format_ax(ax3, xstart, xend, ig, panel_idx=2*4+ig, ylims=(1e-6,1000), ylabel=r"Discharge ($\mathrm{m^3\,s^{-1}}$)" if ig == 0 else "", draw_legend=False, fontsize=annotate_fs*0.9, y_panellabel=6/7)
        if (run_index == run_indices[-1]) & (ig==n_glaciers-1):
            lax = fig.add_subplot(gs[5, 3])
            lax.plot([], [], color='grey', linestyle="dashed", label=r"$Q$"+" (channels)")
            lax.plot([], [], color='grey', linestyle="solid", label=r"$q$"+" (sheets)")
            lax.legend(loc=(0.4,0.0), title="Discharge", frameon=False, handlelength=1.5)
            lax.axis("off")

# Share y-axis for all velocity plots (first row)
all_ylims = [ax.get_ylim() for ax in axes[0,:]]
global_ymin = min([ylim[0] for ylim in all_ylims])
global_ymax = max([ylim[1] for ylim in all_ylims])
for ax in axes[0,:]:
    ax.set_ylim(global_ymin, global_ymax)

plt.tight_layout(pad=0.4)
plt.savefig(f"plotting/main_figures/{figname}.jpg")
