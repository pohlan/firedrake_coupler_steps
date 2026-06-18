import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cmcrameri.cm as cmc

# which glaciers and runs to plot
run_index1, run_index2 = 190, 200
# run_names = ["low k_s", "high k_s"]  # use these instead of M0, M1
glacier_ids = [1,4]

# timeseries_path = f"test_inversion/forward_beta2_{beta2}e5/time_series.h5"
flowlines_path = "Greenland_data/russel/flowlines.gpkg"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"

# profile width and interval
profile_width = 2e3
ds = 2e3   # interval
profile_stop = 50e3
glacier_names = ["Isunnguata Sermia", "Isorlersuup"]
ss = np.arange(start=ds/2, stop=profile_stop,step=ds)

# time period
year = 2017
xstart,xend = datetime(year,1,2), datetime(year,12,31)

# plotting parameters
color_map = cmc.managua
plt.rcParams['font.size'] = 25
annotate_fs = 25
lw = 2.5
clims = (10,250)

# load model stuff not run-specific
timeseries_path = f"parameter_runs/run_{run_index1}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
B, H, S = load_topography(mesh_)

# prepare figure, gridspecs
fig = plt.figure(figsize=(len(glacier_ids)*8,14))
gs = fig.add_gridspec(4, len(glacier_ids)+1, left=0.22, right=0.84, top=0.95, bottom=0.08, hspace=0.15, wspace=0.1, width_ratios=[15, 15, 1], height_ratios=[1, 1, 1, 1])

for (ig,(gl,gl_name)) in enumerate(zip(glacier_ids,glacier_names)):
    fl = [0,4,1,2,3][gl-1]  # flow linestrings don't have the same numbering as gl; also gl starts at 1
    s, s_sub = get_s_functions(flowlines_path, fl, mesh_, smesh_, profile_width)

    # #########################
    # Panel: Model -- #
    # #########################

    for (i_run,run_index) in enumerate([run_index1,run_index2]):
        timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"

        # load model output from hdf5 file
        us_raw, m_raw, phi_raw, q_raw, Q_raw, n_idx = load_model_output(timeseries_path)
        # get model time series
        q, Q, Us, pw_pi, m = get_timestamps(mesh_, smesh_, B, H, us_raw, phi_raw, q_raw, Q_raw, m_raw, n_idx)
        start_date = datetime(2016, 1, 1)
        dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
        i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]

        # organize into matrix
        model_matrix = np.zeros((len(ss),len(i_model)))  # (space,time)
        for (i_space,splus) in enumerate(ss):
            for (i_time,Umod) in enumerate(np.array(Us)[i_model]):
                model_matrix[i_space,i_time] = get_variable(Umod, mesh_, s, [splus-ds/2,splus+ds/2])[0]

        # convert to dataframe in order to calculate monthly means
        dates = pd.to_datetime(np.array(dates_model)[i_model])
        df = pd.DataFrame(
            model_matrix.T,  # shape (time, space)
            index=dates,
            columns=ss
        )
        monthly_means = df.resample('MS').mean()   # month-start frequency
        # make the corresponding date the middle of the month rather than the beginnning, to be consistent with observations
        monthly_means.index = (monthly_means.index + (monthly_means.index + pd.offsets.MonthEnd(0) - monthly_means.index) / 2)
        # plot
        axi = fig.add_subplot(gs[i_run, ig])
        im1 = axi.pcolormesh(
            ss * 1e-3,
            monthly_means.index,
            monthly_means.values,
            cmap=color_map
        )
        # not monthly means, original 2-day timestep
        # im1 = axi.pcolormesh(
        #     ss * 1e-3,
        #     dates,
        #     model_matrix.transpose(),
        #     cmap=color_map
        # )

        im1.set_clim(clims)
        axi.tick_params(axis='x', labelbottom=False, length=4)  # remove x-axis labels
        if ig == 0:
            axi.set_ylabel(f"Month of {year}")
            axi.annotate(f"M{i_run}", xy=[-0.5,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        else:
            axi.tick_params(axis='y', labelleft=False, length=4)  # remove y-axis labels
            cax = fig.add_subplot(gs[i_run, ig+1])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("Speed (m/yr)")
            # fig.colorbar(im1, ax=axi, label="Surface speed (m/yr)")
        if i_run == 0:
            axi.annotate(gl_name, xy=[0.5,1.1], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        axi.set_xlim(0,np.max(ss)*1e-3)
        axi.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axi.yaxis.set_major_formatter(mdates.DateFormatter('%m'))
        axi.annotate(f"{idx_to_letter(i_run*len(glacier_ids)+ig)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.8, fontweight="bold")


    ##########################
    # Panel: Observations -- #
    ##########################

    ax3 = fig.add_subplot(gs[2, ig])

    # load velocity observations
    files   = os.listdir(vel_dir)
    sorted_dates, U_obs, U_mask, _ = load_vel_obs(vel_dir, files, mesh_, xrange=(xstart,xend))
    n_tsteps = len(sorted_dates)

    # organize into matrix
    obs_matrix = np.zeros((len(ss),n_tsteps))  # (space,time)
    for (i_space,splus) in enumerate(ss):
        for (i_time, (Uobs,mask)) in enumerate(zip(U_obs, U_mask)):
            obs_matrix[i_space,i_time] = get_variable(Uobs, mesh_, s, [splus-ds/2,splus+ds/2],mask=mask)[0]
    i_obs = np.where( (np.array(sorted_dates) > xstart) &  (np.array(sorted_dates) < xend) )[0]

    # plot
    im = ax3.pcolormesh(ss*1e-3, sorted_dates[i_obs], obs_matrix[:,i_obs].transpose(), cmap=color_map)
    im.set_clim(clims)
    ax3.tick_params(axis='x', labelbottom=False, length=4)  # remove x-axis labels
    if ig == 0:
        ax3.set_ylabel("Month of 2017")
        ax3.annotate(f"Observed", xy=[-0.5,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    else:
        ax3.tick_params(axis='y', labelleft=False, length=4)  # remove y-axis labels
        # colorbar in separate subplot axis
        cax = fig.add_subplot(gs[2, ig+1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Speed (m/yr)")
    ax3.set_xlim(0,np.max(ss)*1e-3)
    xl = ax3.get_xlim()
    ax3.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.yaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax3.annotate(f"{idx_to_letter(2*len(glacier_ids)+ig)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.8, fontweight="bold")

    ######################
    # Panel: geometry -- #
    ######################

    # A bit coarser so it's smoother
    ds_topo = 2e3
    ss_topo = np.arange(start=500, stop=profile_stop+2*ds_topo, step=ds_topo)

    # distance along flowlines
    ax4 = fig.add_subplot(gs[3, ig])
    s0s_avg = (ss_topo[:-1]+ss_topo[1:]) / 2
    d_along = (s0s_avg - s0s_avg[0] ) / 1e3
    Si = np.array(get_variable(S, mesh_, s, ss_topo))
    Bi = np.array(get_variable(B, mesh_, s, ss_topo))
    line1 = ax4.plot(d_along, Si, label="Surface", color="grey", lw=lw)
    line2 = ax4.plot(d_along, Bi, label="Bed", color="Black", lw=lw)
    ax4.set_ylim(-100,1350)
    ax4.set_yticks([0, 400, 800, 1200])
    ax4.set_xlabel("Distance along profile (km)")
    if ig == 0:
        ax4.set_ylabel("Elevation (m)")
    else:
        ax4.tick_params(axis='y', labelleft=False, length=4)  # remove y-axis labels
        cax = fig.add_subplot(gs[3, ig+1])
        handles = [line1[0], line2[0]]
        labels = [h.get_label() for h in handles]
        cax.legend(handles=handles, labels=labels, loc=(0.22,0.25), frameon=False)
        cax.axis("off")
    ax4.set_xlim(xl)
    # ax4.set_xlim(-0.5,(profile_stop+ds/2)*1e-3)
    ax4.annotate(f"{idx_to_letter(3*len(glacier_ids)+ig)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.8, fontweight="bold")

plt.savefig(f"plotting/output_heatmap/runs_{run_index1}_{run_index2}_gl{gl}_obs.jpg")
# plt.savefig(f"plotting/main_figures/f02.jpg")



