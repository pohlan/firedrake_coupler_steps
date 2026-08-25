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

run_indices_maps = [314, 478]
model_labels = ["Baseline", "Reduced sheet flow"]
xstart_winter, xend_winter = datetime(2020,10,1), datetime(2021,4,30)
xstart_plot, xend_plot = datetime(2020, 5, 1), datetime(2021, 6, 20)

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
x0 = np.mean(xcoords)
cols    = ["magenta", "saddlebrown"]

# plotting parameters
vm = 6
plt.rcParams['font.size'] = 20
annotate_fs = 23
marker_size = 115
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

# paths
timeseries_path = f"parameter_runs/run_{run_indices_maps[0]}/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"


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


def plot_model_panel(ax, fig, gs_top, ii, mesh_, run_index, model_label, B, gdf, vm, annotate_fs, xstart, xend, xcoords, ycoords, cols):
    U_diff = get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, start_year=2014)
    cl = tripcolor(U_diff, axes=ax, vmin=-vm, vmax=vm, cmap="BrBG")
    ax.scatter(xcoords, ycoords, marker_size, color="black", marker="x")
    ax.annotate("A", xy=(xcoords[0]+1e4,ycoords[0]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
    ax.annotate("B", xy=(xcoords[1]+1e4,ycoords[1]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")

    if ii == 1:
        cax = fig.add_subplot(gs_top[0, 3])
        fig.colorbar(cl, ax=cax, label="Monthly winter speed \nincrease " + r"$\Delta u$ " + r"($\mathrm{m\,a^{-1}}$)")
        remove_axes(cax)

    remove_axes(ax)
    ax.set_title(model_label, fontsize=annotate_fs, fontweight="bold")
    panel_letter_annotation(ax, ii+1, annotate_fs)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="Domain")
    ax.vlines(x0, ax.get_ylim()[0]+2.4e4, ax.get_ylim()[1]+1e4, color="grey", lw=4, ls="dotted")
    ax.annotate("Fig. 5 \ncross-section", xy=(x0+0.5e4, ax.get_ylim()[0]), fontsize=annotate_fs, ha="center")


param_labels = {"k_s": r"$k_s$", "h_r": r"$h_r$", "m_basal": r"$m_\mathrm{basal}$"}
param_units  = {"k_s": r"$\mathrm{m^{\frac{7}{4}}\,kg^{-\frac{1}{2}}}$", "h_r": r"$\mathrm{m}$", "m_basal": r"$\mathrm{m\,a^{-1}}$"}


gdf = gpd.read_file(outline_path)

# load model file meshes
mesh_, smesh_ = get_meshes(timeseries_path)


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

# mask out values outside of domain
domain_outline = gu.Vector(outline_path)
mask_noglacier   = ~domain_outline.create_mask(raster0)
dU_obs[mask_noglacier.data.ravel()] = np.nan


fig = plt.figure(figsize=(30, 20))
gs  = fig.add_gridspec(5, 1, left=0.01, right=0.99, top=0.92, bottom=0.06, height_ratios=[0.85,0.12,1,0.1,1])

gs_top = gs[0].subgridspec(1, 5, hspace=0.0, wspace=0.04, width_ratios=[1,1,1,0.15,0.2])
ax = fig.add_subplot(gs_top[0,0])
ax2 = fig.add_subplot(gs_top[0,1])
ax3 = fig.add_subplot(gs_top[0,2])


im = ax.pcolormesh(x,np.flip(y), map, vmin=-vm, vmax=vm, cmap="BrBG") #, cmap='viridis')
ax.scatter(xcoords, ycoords, marker_size, color="black", marker="x")
ax.annotate("A", xy=(xcoords[0]+1e4,ycoords[0]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
ax.annotate("B", xy=(xcoords[1]+1e4,ycoords[1]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
remove_axes(ax)

ax.set_title("Observed", fontsize=annotate_fs, fontweight="bold")
ax.annotate(f"{idx_to_letter(0)}", xy=(0.03,1.0), xycoords="axes fraction", fontsize=annotate_fs*0.95, fontweight="bold")

gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')


#################
# MODEL         #
#################

B, H, S = load_topography(mesh_)

for ii, (run_index, ax_model, model_label) in enumerate(zip(run_indices_maps, [ax2, ax3], model_labels)):
    plot_model_panel(ax_model, fig, gs_top, ii, mesh_, run_index, model_label, B, gdf, vm, annotate_fs, xstart_winter, xend_winter, xcoords, ycoords, cols)


##################
# A and B titles #
##################

gs_mid = gs[1].subgridspec(1,8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

topax = fig.add_subplot(gs_mid[0, 1:4])
topax.fill_between([0,1],0,1, color="grey", alpha=0.1)
topax.set_xlim(0,1)
topax.axis("off")
topax.annotate("A", xy=[0.5,0.4], fontsize=annotate_fs*1.3, fontweight="bold", ha="center", va="center")


topax = fig.add_subplot(gs_mid[0, 5:8])
topax.fill_between([0,1],0,1, color="grey", alpha=0.1)
topax.set_xlim(0,1)
topax.axis("off")
topax.annotate("B", xy=[0.5,0.4], fontsize=annotate_fs*1.1, fontweight="bold", ha="center", va="center")


######################
# Sensitivity to k_s #
######################

run_indices_suit = range(601,633)
run_indices_timeseries = [622, 602]

gs_bot = gs[2].subgridspec(1, 8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

ax = fig.add_subplot(gs_bot[0, 1])
plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "k_s", "m_basal", xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]], colors, 3,
                       param_units, param_labels)

ax_A = fig.add_subplot(gs_bot[0, 3])
plot_timeseries_panel(ax_A, mesh_, vel_dir, run_indices_timeseries, "k_s", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[0]]], [[ycoords[0]]], cols, 4,
                      param_units, param_labels)

ax = fig.add_subplot(gs_bot[0, 5])
plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "k_s", "m_basal", xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]], colors, 5,
                       param_units, param_labels, labelloc=(0.02,0.36))

ax_B = fig.add_subplot(gs_bot[0, 7])
plot_timeseries_panel(ax_B, mesh_, vel_dir, run_indices_timeseries, "k_s", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[1]]], [[ycoords[1]]], cols, 6,
                      param_units, param_labels)



######################
# Sensitivity to h_r #
######################

run_indices_suit = range(633,673)
run_indices_timeseries = [656, 650]

gs_bot = gs[4].subgridspec(1, 8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

ax = fig.add_subplot(gs_bot[0, 1])
plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "h_r", "m_basal", xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]], colors, 7,
                       param_units, param_labels, xscale="linear")

ax = fig.add_subplot(gs_bot[0, 3])
plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, "h_r", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[0]]], [[ycoords[0]]], cols, 8,
                      param_units, param_labels, ylims=ax_A.get_ylim())

ax = fig.add_subplot(gs_bot[0, 5])
plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "h_r", "m_basal", xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]], colors, 9,
                       param_units, param_labels, labelloc="lower right", xscale="linear")

ax = fig.add_subplot(gs_bot[0, 7])
plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, "h_r", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[1]]], [[ycoords[1]]], cols, 10,
                      param_units, param_labels, ylims=ax_B.get_ylim())



# save
plt.savefig(f"plotting/main_figures/f04.jpg")
