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
run_index1, run_index2 = 310, 410
model_start_yrs        = [2014, 2014]
model_labels  = ["Baseline", "Reduced \nsheet flow"]
glacier_ids = [1,2,3,4,5]

# timeseries_path = f"test_inversion/forward_beta2_{beta2}e5/time_series.h5"
flowlines_dir = "Greenland_data/profiles/"
outline_GrIS_path = "Greenland_data/gris-outline-imbie-1980_updated_crs.shp"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

# profile width and interval
profile_width = 2e3
delta = 2e3   # interval
profile_stop = 50e3
glacier_names = ["Isunnguata Sermia", "Russel", "Ørkendalen", "Isorlersuup", "Glacier 5"]
ds_upglacier = np.arange(start=delta/2, stop=profile_stop,step=delta)

# time period
year = 2015
xstart,xend = datetime(year,1,2), datetime(year,12,31)

ds_vline = [np.array([10e3,25e3]),[],[],np.array([15e3,25e3]),[]]

# plotting parameters
color_map = cmc.managua
plt.rcParams['font.size'] = 33
annotate_fs = 33
lw = 2.5
clims = (10,250)

# load model stuff not run-specific
timeseries_path = f"parameter_runs/run_{run_index1}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
B, H, S = load_topography(mesh_)

# prepare figure, gridspecs
fig = plt.figure(figsize=(len(glacier_ids)*8,33))
gs = fig.add_gridspec(6, len(glacier_ids)+1, left=0.11, right=0.92, top=0.95, bottom=0.08, hspace=0.15, wspace=0.1, width_ratios=[15, 15, 15, 15, 15, 1], height_ratios=[2.5, 0.2, 1, 1, 1, 0.8])
# gs = fig.add_gridspec(5, len(glacier_ids)+1, left=0.11, right=0.92, top=0.95, bottom=0.08, hspace=0.15, wspace=0.1, width_ratios=[15, 1], height_ratios=[0.6, 1, 1, 1, 0.8])

# plot bed topography
topax = fig.add_subplot(gs[0, 0:4])

def get_topo(file):
    r = gu.Raster(file)
    # delta = r.res[0]*2
    r.crop([-2.4e5, -2.585e6, 0, -2.47e6], inplace=True)
    outline = gu.Vector(outline_path)
    mask   = ~outline.create_mask(r)
    r.set_mask(mask)
    return r
r_bed = get_topo("Greenland_data/BedMachineGreenland-v5_bed_smooth_sig5.nc")

data    = r_bed.data.squeeze()
x, y    = r_bed.coords(grid=False)
im = topax.pcolormesh(x,np.flip(y),data,cmap="terrain")
cbar = fig.colorbar(im, ax=topax)
cbar.set_label("Bed elevation (m)")
# format
topax.set_aspect('equal')
topax.set_xticks([])
topax.set_yticks([])
topax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
topax.set_ylim(-2.585e6,-2.47e6)
topax.set_xlim(-2.4e5,-2e4)
# domain outline
gdf = gpd.read_file(outline_path)
gdf.plot(ax=topax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

######################
# Scalebar & N arrow #
######################

# scale bar
line_x = [-2.35e5,-2.25e5]
line_y = [-2.58e6,-2.58e6]
topax.plot(line_x, line_y, color="black", lw=5)
topax.annotate("10 km", [-2.4e5,line_y[0]+5e3], c="black", fontsize=annotate_fs)
# north arrow
topax.arrow(
    -2.05e5, -2.58e6,
    0, 1.2e4,
    head_width=2500,
    head_length=3000,
    color='black'
)
topax.annotate("N", [-2.02e5,line_y[0]+5e3], c="black", fontsize=annotate_fs)

######################
# Surface elevation  #
######################

r_thick = get_topo("Greenland_data/BedMachineGreenland-v5_thickness_smooth_sig5.nc")
r_surf  = r_bed+r_thick
data    = r_surf.data.squeeze()
x, y    = r_surf.coords(grid=False)
cs = topax.contour(x, np.flip(y), data,
                levels=np.arange(0, 3000, 100),
                colors="black",
                linewidths=0.5,
)
labels = topax.clabel(cs, fmt="%d m", fontsize=28, levels=np.arange(900, 3000, 300), rightside_up=False)

######################
# Flowline profiles  #
######################

gls = [1,2,3,4,5]
ds_timeseries = [np.array([10e3,30e3]),[],[],np.array([15e3,25e3]),[]]
colors = ["coral","cornflowerblue","yellowgreen","palevioletred"]

annotate_offsets = [[-1e4,1e4],[-4e4,-5e3],[-4e4,-5e3],[-3.5e4,-1e4],[-2.7e4,-1e4]]

gl_names = ["Isunnguata-Sermia","Russel","Ørkendalen","Isorlersuup","Glacier-5"]
gl_annotations = ["Isunnguata\n Sermia","Russel","Ørkendalen","Isorlersuup","Glacier 5"]
for (gl,strng,gl_name,offset_crd,d_timeseries) in zip(gls, gl_annotations, gl_names, annotate_offsets, ds_timeseries):
    col = "black"
    fl = [0,4,1,2,3][gl-1]
    dists, xx, yy = get_profile(flowlines_dir, gl_name)
    i_dists = np.where(dists<profile_stop)
    dists, xx, yy = dists[i_dists], xx[i_dists], yy[i_dists]
    topax.plot(xx,yy,color=col,lw=5)

    marker_x = []
    marker_y = []
    d_marker = [10e3, 25e3, 40e3]
    for d in d_marker:
        marker_x.append(xx[np.argmin(abs(dists-d))])
        marker_y.append(yy[np.argmin(abs(dists-d))])
    topax.scatter(marker_x, marker_y, 80, c=col, edgecolors='black', marker="s",zorder=3)
    topax.annotate(strng,[xx[0],yy[0]], xytext=[marker_x[0]+offset_crd[0],marker_y[0]+offset_crd[1]], c=col, fontsize=annotate_fs)

fig.tight_layout()

################
# Panel label  #
################
topax.annotate("a", [-2.4e5,-2.47e6-1e4], xytext=[-2.4e5-2e4,-2.47e6-1e4], fontsize=annotate_fs, fontweight="bold")


######################
# Greenland inset    #
######################

gdf_GrIS = gpd.read_file(outline_GrIS_path)
axins2 = fig.add_subplot(gs[0, 4])
gdf_GrIS.plot(ax=axins2, facecolor='none', edgecolor='black', linewidth=0.5)

gdf.plot(ax=axins2, facecolor='firebrick', alpha=0.8, edgecolor='black', linewidth=2)

axins2.set_aspect('equal')
axins2.set_xticks([])
axins2.set_yticks([])
axins2.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
x0, x1 = axins2.get_xlim()
y0, y1 = axins2.get_ylim()
axins2.set_xlim(x0-0.2*(x1-x0), x1+0.2*(x1-x0))
axins2.set_ylim(y0-0.2*(y1-y0), y1+0.2*(y1-y0))

# arrow
x, y    = r_surf.coords(grid=False)
x0 = np.mean(x)
y0 = np.mean(y)
axins2.arrow(
    x0+3e5, y0+8e5,
    -1.8e5, -5.6e5,
    lw=2.5,
    head_width=100000,
    head_length=90000,
    color='firebrick'
)


for (ig,(gl,gl_name,d_vline)) in enumerate(zip(glacier_ids,glacier_names,ds_vline)):
    fl = [0,4,1,2,3][gl-1]  # flow linestrings don't have the same numbering as gl; also gl starts at 1

    print(gl_name)

    dists, xx, yy = get_profile(flowlines_dir, gl_name)

    xc_prof, yc_prof = [], []
    for (i_space, d) in enumerate(ds_upglacier):
        idx_profiles = np.where(abs(dists - d)<delta/2)
        xc_prof.append(xx[idx_profiles])
        yc_prof.append(yy[idx_profiles])

    # #########################
    # Panel: Model -- #
    # #########################

    for (i_run,(run_index,model_start_yr)) in enumerate(zip([run_index1,run_index2],model_start_yrs)):
        timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"

        # load model output from hdf5 file
        us_raw, m_raw, phi_raw, q_raw, Q_raw, n_idx, _ = load_model_output(timeseries_path)

        # get model time vector
        start_date = datetime(model_start_yr, 1, 1)
        dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
        i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]
        dates_model = np.array(dates_model)[i_model]

        # get model time series, organize into matrix
        # model_matrix = np.zeros((len(ds_upglacier),len(i_model)))  # (space,time)
        print(f"Load model {i_run}...")
        model_matrix = load_model_timeseries(mesh_, us_raw, xc_prof, yc_prof, i_model)

        # convert to dataframe in order to calculate monthly means
        dates = pd.to_datetime(dates_model)
        df = pd.DataFrame(
            model_matrix,  # shape (time, space)
            index=dates,
            columns=ds_upglacier
        )
        monthly_means = df.resample('MS').mean()   # month-start frequency
        # make the corresponding date the middle of the month rather than the beginnning, to be consistent with observations
        monthly_means.index = (monthly_means.index + (monthly_means.index + pd.offsets.MonthEnd(0) - monthly_means.index) / 2)
        # plot
        axi = fig.add_subplot(gs[i_run+2, ig])
        im1 = axi.pcolormesh(
            ds_upglacier * 1e-3,
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

        if len(d_vline) > 0:
            axi.vlines(x=d_vline*1e-3, ymin=axi.get_ylim()[0], ymax=axi.get_ylim()[1], color='cyan', linestyle='-', lw=3)


        im1.set_clim(clims)
        axi.tick_params(axis='x', labelbottom=False, length=4)  # remove x-axis labels
        if ig == 0:
            axi.set_ylabel(f"Month of {year}")
            # axi.annotate(f"M{i_run}", xy=[-0.5,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
            axi.annotate(model_labels[i_run], xy=[-0.5,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        else:
            axi.tick_params(axis='y', labelleft=False, length=4)  # remove y-axis labels
        if ig == len(glacier_ids)-1:
            cax = fig.add_subplot(gs[i_run+2, ig+1])
            cbar = fig.colorbar(im1, cax=cax)
            cbar.set_label("Speed (m/yr)")
            # fig.colorbar(im1, ax=axi, label="Surface speed (m/yr)")
        if i_run == 0:
            axi.annotate(gl_name, xy=[0.5,1.1], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        axi.set_xlim(0,np.max(ds_upglacier)*1e-3)
        axi.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axi.yaxis.set_major_formatter(mdates.DateFormatter('%m'))
        axi.annotate(f"{idx_to_letter(i_run*len(glacier_ids)+ig)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.8, fontweight="bold")


    ##########################
    # Panel: Observations -- #
    ##########################

    ax3 = fig.add_subplot(gs[4, ig])

    # load velocity observations
    files   = os.listdir(vel_dir)
    obs_dates, obs_files = get_obs_files(files, xrange=(xstart,xend))

    # organize into matrix
    # obs_matrix = np.zeros((len(ds_upglacier),len(obs_dates)))  # (space,time)
    # for (i_space, d) in enumerate(ds_upglacier):
        # idx_profiles = np.where(abs(dists - d)<delta/2)
        # obs_matrix[i_space,:] = np.mean(

    obs_matrix = load_obs_timeseries(vel_dir, obs_files, xc_prof, yc_prof)

    # plot
    im = ax3.pcolormesh(ds_upglacier*1e-3, obs_dates, obs_matrix, cmap=color_map)
    im.set_clim(clims)
    ax3.tick_params(axis='x', labelbottom=False, length=4)  # remove x-axis labels
    if ig == 0:
        ax3.set_ylabel(f"Month of {year}")
        ax3.annotate(f"Observed", xy=[-0.5,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    else:
        ax3.tick_params(axis='y', labelleft=False, length=4)  # remove y-axis labels
    if ig == len(glacier_ids)-1:
        # colorbar in separate subplot axis
        cax = fig.add_subplot(gs[4, ig+1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Speed (m/yr)")
    ax3.set_xlim(0,np.max(ds_upglacier)*1e-3)
    xl = ax3.get_xlim()
    ax3.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.yaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax3.annotate(f"{idx_to_letter(2*len(glacier_ids)+ig)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.8, fontweight="bold")

    if len(d_vline) > 0:
            ax3.vlines(x=d_vline*1e-3, ymin=ax3.get_ylim()[0], ymax=ax3.get_ylim()[1], color='cyan', linestyle='-', lw=3)



    ######################
    # Panel: geometry -- #
    ######################

    # A bit coarser so it's smoother
    delta_topo = 2e3
    ss_topo = np.arange(start=500, stop=profile_stop+2*delta_topo, step=delta_topo)

    # distance along flowlines
    ax4 = fig.add_subplot(gs[5, ig])
    # s0s_avg = (ss_topo[:-1]+ss_topo[1:]) / 2
    # d_along = (s0s_avg - s0s_avg[0] ) / 1e3
    S_profile = np.zeros(len(ss_topo))
    B_profile = np.zeros(len(ss_topo))
    for (i_space, d) in enumerate(ss_topo):
        idx_profiles = np.where(abs(dists - d)<delta_topo/2)
        S_profile[i_space] = np.mean(interpolate_meshfct_to_profile(S, xx[idx_profiles], yy[idx_profiles]))
        B_profile[i_space] = np.mean(interpolate_meshfct_to_profile(B, xx[idx_profiles], yy[idx_profiles]))
    line1 = ax4.plot(ss_topo*1e-3, S_profile, label="Surface", color="grey", lw=lw)
    line2 = ax4.plot(ss_topo*1e-3, B_profile, label="Bed", color="Black", lw=lw)
    ax4.set_ylim(-180,1350)
    ax4.set_yticks([0, 400, 800, 1200])
    ax4.set_xlabel("Distance along profile (km)")
    if ig == 0:
        ax4.set_ylabel("Elevation (m)")
    else:
        ax4.tick_params(axis='y', labelleft=False, length=4)  # remove y-axis labels
        cax = fig.add_subplot(gs[5, ig+1])
        handles = [line1[0], line2[0]]
        labels = [h.get_label() for h in handles]
        cax.legend(handles=handles, labels=labels, loc=(0.22,0.25), frameon=False)
        cax.axis("off")
    ax4.set_xlim(xl)
    # ax4.set_xlim(-0.5,(profile_stop+ds/2)*1e-3)
    ax4.annotate(f"{idx_to_letter(3*len(glacier_ids)+ig)}", xy=(0.03,0.9), xycoords="axes fraction", fontsize=annotate_fs*0.8, fontweight="bold")

    if len(d_vline) > 0:
            ax4.vlines(x=d_vline*1e-3, ymin=ax4.get_ylim()[0], ymax=ax4.get_ylim()[1], color='cyan', linestyle='-', lw=3)


plt.savefig(f"plotting/output_heatmap/hovmuller_runs_{run_index1}_{run_index2}_{year}.jpg")
# plt.savefig(f"plotting/main_figures/f02.jpg")
