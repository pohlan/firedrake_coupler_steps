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


years_0 = range(2019,2024)

run_indices_maps = [314, 478]
model_labels = ["Baseline", "Reduced sheet flow"]

# plotting parameters
vm = 10
plt.rcParams['font.size'] = 20
annotate_fs = 24
marker_size = 115
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

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

def get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, start_year=2014):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, _, _, _, _, _, _, n_idx, _ = load_model_output(timeseries_path)

    dates_model, i_model = get_model_dates(n_idx, xstart, xend, start_year=start_year)
    dates_model_pd = pd.to_datetime(dates_model)

    dU = np.zeros(us_raw.shape[0])
    for n in range(us_raw.shape[0]):
        U_model = us_raw[n, i_model]
        df_monthly = pd.DataFrame(U_model, index=dates_model_pd)
        U_monthly = df_monthly.resample("MS").mean().to_numpy().ravel()
        dU[n] = fit_linear_slope(U_monthly)

    U_diff = B
    U_diff.dat.data[:] = dU # monthly_means.diff(axis=0).median(axis=0)
    return U_diff

def plot_model_panel(ax, fig, gs_top, ii, iy, mesh_, run_index, B, gdf, vm, annotate_fs, xstart, xend):
    U_diff = get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, start_year=2014)
    cl = tripcolor(U_diff, axes=ax, vmin=-vm, vmax=vm, cmap="BrBG")

    if ii == 1:
        cax = fig.add_subplot(gs_top[2, 3])
        fig.colorbar(cl, ax=cax, label="Monthly winter speed \nincrease " + r"$\Delta u$ " + r"($\mathrm{m\,a^{-1}}$)")
        remove_axes(cax)

    remove_axes(ax)
    # if iy == 0:
    #     ax.set_title(model_label, fontsize=annotate_fs, fontweight="bold")
    panel_letter_annotation(ax, 3*iy+ii+1, annotate_fs)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="Domain")


def get_obs_speedup(xstart_winter,xend_winter):
    # load velocity observations
    files = glob.glob(vel_dir+"*vv*.tif")
    obs_dates, obs_files = get_obs_files(files, xrange=(xstart_winter,xend_winter))
    ex_files, ey_files, vx_files, vy_files = get_obs_error_files(xstart_winter, xend_winter, vel_dir)

    raster0 = get_raster(obs_files[0])
    nx, ny = raster0.data.shape
    Uobs_time = np.zeros((raster0.data.size, len(obs_files)))
    sigma_time = np.zeros((raster0.data.size, len(obs_files)))
    for (n,(fvv,fex,fey,fvx,fvy)) in enumerate(zip(obs_files,ex_files,ey_files,vx_files,vy_files)):
        raster = get_raster(fvv)
        Uobs_time[:,n] = raster.data.ravel()
        e_x = get_raster(fex).data.ravel()
        e_y = get_raster(fey).data.ravel()
        v_x = get_raster(fvx).data.ravel()
        v_y = get_raster(fvy).data.ravel()
        sigma_time[:,n] = obs_sigma(v_x, v_y, e_x, e_y)

    dU_obs = np.zeros(raster0.data.size)
    for i in range(Uobs_time.shape[0]):
        slope, slope_err = fit_linear_slope(Uobs_time[i,:], sigma_time[i,:])
        dU_obs[i] = slope
    map  = dU_obs.reshape(nx,ny)
    x, y = raster0.coords(grid=False)
    mask_noglacier   = ~domain_outline.create_mask(raster0)
    dU_obs[mask_noglacier.data.ravel()] = np.nan
    dU_obs[dU_obs==1.0] = np.nan
    return x, y, map



fig = plt.figure(figsize=(30, 5*len(years_0)))
gs  = fig.add_gridspec(len(years_0), 5, left=0.1, right=0.99, top=0.94, bottom=0.01, width_ratios=[1,1,1,0.15,0.2])

for (iy,yr) in enumerate(years_0):
    ax = fig.add_subplot(gs[iy,0])
    ax2 = fig.add_subplot(gs[iy,1])
    ax3 = fig.add_subplot(gs[iy,2])

    xstart_winter, xend_winter = datetime(yr,10,1), datetime(yr+1,4,30)

    x, y, map = get_obs_speedup(xstart_winter, xend_winter)
    im = ax.pcolormesh(x,np.flip(y), map, vmin=-vm, vmax=vm, cmap="BrBG") #, cmap='viridis')
    remove_axes(ax)

    ax.annotate(f"{yr}/{yr+1}", xy=[-0.25,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    if iy == 0:
        ax.annotate("Observed", xy=[0.5,1.25], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    ax.annotate(f"{idx_to_letter(iy*3)}", xy=(0.03,1.0), xycoords="axes fraction", fontsize=annotate_fs*0.95, fontweight="bold")
    gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

    for ii, (run_index, ax_model, model_label) in enumerate(zip(run_indices_maps, [ax2, ax3], model_labels)):
        plot_model_panel(ax_model, fig, gs, ii, iy, mesh_, run_index, B, gdf, vm, annotate_fs, xstart_winter, xend_winter)
        if iy == 0:
            ax_model.annotate(model_label, xy=[0.5,1.25], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")




plt.savefig(f"plotting/main_figures/fS04.jpg")
