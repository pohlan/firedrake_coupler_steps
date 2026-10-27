import os
import sys
import glob
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from firedrake.pyplot import tripcolor
import pandas as pd
import cmcrameri.cm as cmc

years_0 = range(2019,2024)

run_indices_maps = [312, 706]
model_labels = ["Baseline", "Reduced sheet flow"]

# plotting parameters
vm = 10
vmin_s, vmax_s = 0, 110
plt.rcParams['font.size'] = 33
annotate_fs = 33
marker_size = 115
cmap_summer = cmc.nuuk

# paths
timeseries_path = f"parameter_runs/run_{run_indices_maps[0]}/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

# load outline
gdf = gpd.read_file(outline_path)
domain_outline = gu.Vector(outline_path)

# load model file meshes
mesh_, smesh_ = get_meshes(timeseries_path)
B, H, S = load_topography(mesh_)
V = df.FunctionSpace(mesh_, "CG", 1)

def get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, xstart_summer, xend_summer, start_year=2014):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, _, _, _, _, _, _, n_idx, _ = load_model_output(timeseries_path)

    dates_model_winter, i_model_winter = get_model_dates(n_idx, xstart, xend, start_year=start_year)
    dates_model_pd_winter = pd.to_datetime(dates_model_winter)
    dates_model_summer, i_model_summer = get_model_dates(n_idx, xstart_summer, xend_summer)
    dates_model_pd_summer = pd.to_datetime(dates_model_summer)

    dU_winter = np.zeros(us_raw.shape[0])
    dU_summer = np.zeros(us_raw.shape[0])
    for n in range(us_raw.shape[0]):
        U_model = us_raw[n, i_model_winter]
        df_monthly = pd.DataFrame(U_model, index=dates_model_pd_winter)
        U_monthly_winter = df_monthly.resample("MS").mean().to_numpy().ravel()
        dU_winter[n] = fit_linear_slope(U_monthly_winter)

        U_model = us_raw[n, i_model_summer]
        df_monthly = pd.DataFrame(U_model, index=dates_model_pd_summer)
        U_monthly_summer = df_monthly.resample("MS").mean().to_numpy().ravel()
        dU_summer[n] = (np.max(U_monthly_summer) - np.mean(np.concatenate([U_monthly_summer,U_monthly_winter])))

    U_diff_winter = df.Function(V)
    U_diff_winter.dat.data[:] = dU_winter

    U_diff_summer = df.Function(V)
    U_diff_summer.dat.data[:] = dU_summer

    return U_diff_winter, U_diff_summer

def plot_model_panel(ax_w, ax_s, fig_w, fig_s, gs_top_w, gs_top_s, ii, iy, mesh_, run_index, B, gdf, vm, annotate_fs, xstart, xend, xstart_summer, xend_summer):
    U_diff_w, U_diff_s = get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, xstart_summer, xend_summer, start_year=2014)

    cl = tripcolor(U_diff_w, axes=ax_w, vmin=-vm, vmax=vm, cmap="BrBG")
    cl = tripcolor(U_diff_s, axes=ax_s, vmin=vmin_s, vmax=vmax_s, cmap=cmap_summer)

    if ii == 1:
        cax = fig_w.add_subplot(gs_top_w[2, 3])
        fig_w.colorbar(cl, cax=cax, label=r"$\dot{u}_\mathrm{winter}$ " + r"($\mathrm{m\,a^{-1}/month}$)")
        pos = cax.get_position()
        cax.set_position([
            pos.x0 + 0 * pos.width,
            pos.y0 - 0.05 * pos.height,
            0.15*pos.width,
            1.1*pos.height
        ])
        cax = fig_s.add_subplot(gs_top_s[2, 3])
        fig_s.colorbar(cl, cax=cax, label=r"$\Delta u_\mathrm{max}$ " + r"$\mathrm{(m\,a^{-1})}$")
        pos = cax.get_position()
        cax.set_position([
            pos.x0 + 0 * pos.width,
            pos.y0 - 0.05 * pos.height,
            0.15*pos.width,
            1.1*pos.height
        ])

    for ax in [ax_w,ax_s]:
        remove_axes(ax)
        panel_letter_annotation(ax, 3*iy+ii+1, annotate_fs)
        gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="Domain")


def get_obs_speedup(xstart_summer, xend_summer, xstart_winter,xend_winter):
    # load velocity observations
    files = glob.glob(vel_dir+"*vv*.tif")
    obs_dates_winter, obs_files_winter = get_obs_files(files, xrange=(xstart_winter,xend_winter))
    ex_files, ey_files, vx_files, vy_files = get_obs_error_files(xstart_winter, xend_winter, vel_dir)
    obs_dates_summer, obs_files_summer = get_obs_files(files, xrange=(xstart_summer,xend_summer))

    raster0 = get_raster(obs_files_winter[0])
    nx, ny = raster0.data.shape
    Uobs_time_winter = np.zeros((raster0.data.size, len(obs_files_winter)))
    sigma_time = np.zeros((raster0.data.size, len(obs_files_winter)))
    for (n,(fvv,fex,fey,fvx,fvy)) in enumerate(zip(obs_files_winter,ex_files,ey_files,vx_files,vy_files)):
        raster = get_raster(fvv)
        Uobs_time_winter[:,n] = raster.data.ravel()
        e_x = get_raster(fex).data.ravel()
        e_y = get_raster(fey).data.ravel()
        v_x = get_raster(fvx).data.ravel()
        v_y = get_raster(fvy).data.ravel()
        sigma_time[:,n] = obs_sigma(v_x, v_y, e_x, e_y)

    Uobs_time_summer = np.zeros((raster0.data.size, len(obs_files_summer)))
    for (n,fvv) in enumerate(obs_files_summer):
        raster = get_raster(fvv)
        Uobs_time_summer[:,n] = raster.data.ravel()

    dU_obs_winter = np.zeros(raster0.data.size)
    dU_obs_summer = np.zeros(raster0.data.size)
    for i in range(Uobs_time_summer.shape[0]):
        slope, slope_err = fit_linear_slope(Uobs_time_winter[i,:], sigma_time[i,:])
        dU_obs_winter[i] = slope
        if (all(np.isfinite(Uobs_time_summer[i,:]))) & (all(Uobs_time_summer[i,:] > 0)):
            dU_obs_summer[i] = (np.max(Uobs_time_summer[i,:]) - np.mean(np.concatenate([Uobs_time_summer[i,:], Uobs_time_winter[i,:]])))
        else:
            dU_obs_summer[i] = np.nan

    x, y = raster0.coords(grid=False)
    mask_noglacier   = ~domain_outline.create_mask(raster0)
    dU_obs_winter[mask_noglacier.data.ravel()] = np.nan
    dU_obs_summer[mask_noglacier.data.ravel()] = np.nan
    map_w  = dU_obs_winter.reshape(nx,ny)
    # map_w[map_w==1.0] = np.nan
    map_s  = dU_obs_summer.reshape(nx,ny)
    map_s[map_s==0] = np.nan

    return x, y, map_w, map_s


# winter speedup figure
fig_w = plt.figure(figsize=(20, 3*len(years_0)))
gs_w  = fig_w.add_gridspec(len(years_0), 4, left=0.1, wspace=0, hspace=0, right=1.0, top=0.93, bottom=0.01, width_ratios=[1,1,1,0.4])

# summer speedup figure
fig_s = plt.figure(figsize=(20, 3*len(years_0)))
gs_s  = fig_w.add_gridspec(len(years_0), 4, left=0.1, wspace=0, hspace=0, right=1.0, top=0.93, bottom=0.01, width_ratios=[1,1,1,0.4])

for (iy,yr) in enumerate(years_0):
    xstart_summer, xend_summer = datetime(yr,5,1), datetime(yr,9,30)
    xstart_winter, xend_winter = datetime(yr,10,1), datetime(yr+1,4,30)
    x, y, map_w, map_s = get_obs_speedup(xstart_summer, xend_summer, xstart_winter,xend_winter)
    # winter
    ax1_w = fig_w.add_subplot(gs_w[iy,0])
    im = ax1_w.pcolormesh(x,np.flip(y), map_w, vmin=-vm, vmax=vm, cmap="BrBG")
    # summer
    ax1_s = fig_s.add_subplot(gs_s[iy,0])
    im = ax1_s.pcolormesh(x,np.flip(y), map_s, vmin=vmin_s, vmax=vmax_s, cmap=cmap_summer)

    for ax in [ax1_w, ax1_s]:
        remove_axes(ax)
        ax.annotate(f"{yr}/{yr+1-2000}", xy=[-0.18,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        if iy == 0:
            ax.annotate("Observed", xy=[0.5,1.25], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        ax.annotate(f"{idx_to_letter(iy*3)}", xy=(0.03,1.0), xycoords="axes fraction", fontsize=annotate_fs*0.95, fontweight="bold")
        gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

    for ii, (run_index, model_label) in enumerate(zip(run_indices_maps, model_labels)):
        ax_w = fig_w.add_subplot(gs_w[iy,ii+1])
        ax_s = fig_s.add_subplot(gs_s[iy,ii+1])
        plot_model_panel(ax_w, ax_s, fig_w, fig_s, gs_w, gs_s, ii, iy, mesh_, run_index, B, gdf, vm, annotate_fs, xstart_winter, xend_winter, xstart_summer, xend_summer)
        if iy == 0:
            for ax in [ax_w,ax_s]:
                ax.annotate(model_label, xy=[0.5,1.25], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")


fig_w.savefig(f"plotting/main_figures/fS06.jpg")
fig_s.savefig(f"plotting/main_figures/fS07.jpg")
