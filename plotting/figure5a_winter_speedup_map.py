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
from cycler import cycler

run_indices_suit = range(601,629)
run_indices_timeseries = [327, 477]
# run_indices_timeseries = [626, 610]
model_labels = ["Baseline", "Reduced sheet flow"]
xstart,xend = datetime(2020,10,1),datetime(2021,4,30)

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
cols    = ["magenta", "saddlebrown"]

vm = 6

plt.rcParams['font.size'] = 20
annotate_fs = 23
marker_size = 115
lw_times = 2
marker_size_times = 8

# paths
timeseries_path = f"parameter_runs/run_{run_indices_timeseries[0]}/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"


def get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, start_year=2014):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, _, _, _, _, _, n_idx, _ = load_model_output(timeseries_path)

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
        fig.colorbar(cl, ax=cax, label="Monthly speed \nincrease " + r"$\Delta u$ " + r"($\mathrm{m\,a^{-1}}$)")
        remove_axes(cax)

    remove_axes(ax)
    ax.set_title(model_label, fontsize=annotate_fs, fontweight="bold")
    panel_letter_annotation(ax, ii+1, annotate_fs)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="Domain")

def get_obs_error_files(xstart, xend):
    _, ex_files = get_obs_files(glob.glob(vel_dir+"*ex*.tif"), xrange=(xstart,xend))
    _, ey_files = get_obs_files(glob.glob(vel_dir+"*ey*.tif"), xrange=(xstart,xend))
    _, vx_files = get_obs_files(glob.glob(vel_dir+"*vx*.tif"), xrange=(xstart,xend))
    _, vy_files = get_obs_files(glob.glob(vel_dir+"*vy*.tif"), xrange=(xstart,xend))
    return ex_files, ey_files, vx_files, vy_files

def obs_sigma(v_x, v_y, e_x, e_y):
    return np.sqrt(v_x**2*e_x**2+v_y**2*e_y**2) / np.sqrt(v_x**2+v_y**2)

def get_obs_errors(xstart, xend, xc, yc):
    ex_files, ey_files, vx_files, vy_files = get_obs_error_files(xstart, xend)
    e_x = load_obs_timeseries(ex_files, xc, yc)[:, 0]
    e_y = load_obs_timeseries(ey_files, xc, yc)[:, 0]
    v_x = load_obs_timeseries(vx_files, xc, yc)[:, 0]
    v_y = load_obs_timeseries(vy_files, xc, yc)[:, 0]
    sig = obs_sigma(v_x, v_y, e_x, e_y)
    return sig

# def get_obs_errors_raster():
#     ex_files, ey_files, vx_files, vy_files = get_obs_error_files(xstart, xend)
#     raster = get_raster(f)


def plot_sensitivity_panel(ax, mesh_, run_indices, var1, var2, xstart, xend, xc, yc, colors, ii, param_units, param_labels, xscale="log", labelloc="best"):
    dUs = []
    param1 = []
    param2 = []

    for run_idx in run_indices:
        try:
            U_monthly = get_monthly_means_model(mesh_, run_idx, xstart, xend, xc, yc, start_year=2014)
            if len(U_monthly) == 0:
                continue
            dU = fit_linear_slope(U_monthly)
            dUs.append(dU)
            float_params = hlp.get_params_from_input_file(run_idx)
            param1.append(float_params[var1])
            param2.append(float_params[var2])
            print(f"run {run_idx}, {var1} = {float_params[var1]}")
        except:
            continue

    param1 = np.array(param1)
    dUs = np.array(dUs)

    obs_files = glob.glob(vel_dir+"*vv*.tif")
    obs_dates, obs_files = get_obs_files(obs_files, xrange=(xstart,xend))
    Uobs_glaciers = load_obs_timeseries(obs_files, xc, yc)[:, 0]
    # obs_m = np.mean(np.diff(Uobs_glaciers))
    sig = get_obs_errors(xstart, xend, xc, yc)
    slope, slope_err = fit_linear_slope(Uobs_glaciers, sig)

    ax.set_prop_cycle(cycler(color=colors))
    for m in np.unique(param2):
        i_m = np.where(param2 == m)
        idx = np.argsort(param1[i_m])
        ax.plot(param1[i_m][idx], dUs[i_m][idx], label=f"{m}", marker="o", lw=lw_times, markersize=marker_size_times)

    xmin, xmax = ax.get_xlim()
    ax.fill_between([xmin,xmax], slope-slope_err, slope+slope_err, alpha=0.1, color="black")
    ax.hlines(slope, xmin, xmax, color="black", ls="dashed", lw=lw_times)
    # ax.legend(title=r"$m_\mathrm{basal}$ ($\mathrm{m\,a^{-1}}$)")
    ax.legend(title=param_labels[var2]+" ("+ param_units[var2] +")", loc=labelloc)
    ax.set_xlabel(param_labels[var1]+" ("+ param_units[var1] +")")
    ax.set_ylabel(r"$\Delta u$ ($\mathrm{m\,a^{-1}}$)")
    ax.set_xscale(xscale)
    panel_letter_annotation(ax, ii+3, annotate_fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    y0, y1 = ax.get_ylim()
    ax.annotate("Observed", xy=(1.2e-3, slope+slope_err+0.02*(y1-y0)), fontsize=annotate_fs * 0.9)


def plot_timeseries_panel(ax, mesh_, run_indices, var, xstart_plot, xend_plot, xc, yc, cols, ii, param_units, param_labels):
    obs_dates, obs_files = get_obs_files(files, xrange=(xstart_plot, xend_plot))
    for (run_idx,c) in zip(run_indices,cols):
        float_params = hlp.get_params_from_input_file(run_idx)
        param = float_params[var]
        # m  = float_params["m_basal"]
        dates_model, U_model = get_model_timeseries_for_locations(
            mesh_, run_idx, xstart_plot, xend_plot, xc, yc, start_year=2014
        )
        lab = param_labels[var] + f" = {param} " + param_units[var]
        ax.plot(dates_model, U_model - np.mean(U_model), color=c, label=lab, lw=lw_times) #+"\n"+rf"$m_\mathrm{{basal}}= {m}$")

    Uobs_glaciers = load_obs_timeseries(obs_files, xc, yc)[:, 0]
    ax.plot(obs_dates, Uobs_glaciers - np.nanmean(Uobs_glaciers), color="black", ls="dashed", marker="o", label="Observed", lw=lw_times, markersize=marker_size_times)
    ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], xstart, xend, alpha=0.15, facecolor="cornflowerblue")
    ax.set_xlabel("Month of winter 2020/21")
    ax.set_ylabel(r"Speed rel. to mean ($\mathrm{m\,a^{-1}}$)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    panel_letter_annotation(ax, ii+3, annotate_fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

param_labels = {"k_s": r"$k_s$", "m_basal": r"$m_\mathrm{basal}$"}
param_units  = {"k_s": r"$\mathrm{m^{\frac{7}{4}}\,kg^{-\frac{1}{2}}}$", "m_basal": r"$\mathrm{m\,a^{-1}}$"}



gdf = gpd.read_file(outline_path)

# load model file meshes
mesh_, smesh_ = get_meshes(timeseries_path)


# load velocity observations
files = glob.glob(vel_dir+"*vv*.tif")
obs_dates, obs_files = get_obs_files(files, xrange=(xstart,xend))
ex_files, ey_files, vx_files, vy_files = get_obs_error_files(xstart, xend)

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


fig = plt.figure(figsize=(30, 13))
gs  = fig.add_gridspec(4, 1, left=0.01, right=0.99, top=0.92, bottom=0.1, height_ratios=[0.85,0.0,0.12,1])

gs_top = gs[0].subgridspec(1, 5, hspace=0.0, wspace=0.04, width_ratios=[1,1,1,0.15,0.2])
ax = fig.add_subplot(gs_top[0,0])
ax2 = fig.add_subplot(gs_top[0,1])
ax3 = fig.add_subplot(gs_top[0,2])


im = ax.pcolormesh(x,np.flip(y),map, vmin=-vm, vmax=vm, cmap="BrBG") #, cmap='viridis')
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

for ii, (run_index, ax_model, model_label) in enumerate(zip(run_indices_timeseries, [ax2, ax3], model_labels)):
    plot_model_panel(ax_model, fig, gs_top, ii, mesh_, run_index, model_label, B, gdf, vm, annotate_fs, xstart, xend, xcoords, ycoords, cols)


######################
# Sensitivity to k_s #
######################

colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

flowlines_dir = "Greenland_data/profiles/"
gl_name = "Isorlersuup"
delta = 500
d = 15e3

gs_mid = gs[2].subgridspec(1,8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

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

# obs_dates_profile, obs_files_profile = get_obs_files(files, xrange=(xstart, xend))
# obs_dates_extended, obs_files_extended = get_obs_files(files, xrange=(datetime(2020, 9, 1), datetime(2021, 6, 10)))

gs_bot = gs[3].subgridspec(1, 8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])



ax = fig.add_subplot(gs_bot[0, 1])
plot_sensitivity_panel(ax, mesh_, run_indices_suit, "k_s", "m_basal", xstart, xend, [[xcoords[0]]], [[ycoords[0]]], colors, 0,
                       param_units, param_labels)

ax = fig.add_subplot(gs_bot[0, 3])
plot_timeseries_panel(ax, mesh_, run_indices_timeseries, "k_s", datetime(2020, 9, 1), datetime(2021, 6, 10), [[xcoords[0]]], [[ycoords[0]]], cols, 1,
                      param_units, param_labels)

ax = fig.add_subplot(gs_bot[0, 5])
plot_sensitivity_panel(ax, mesh_, run_indices_suit, "k_s", "m_basal", xstart, xend, [[xcoords[1]]], [[ycoords[1]]], colors, 2,
                       param_units, param_labels, labelloc=(0.02,0.36))

ax = fig.add_subplot(gs_bot[0, 7])
plot_timeseries_panel(ax, mesh_, run_indices_timeseries, "k_s", datetime(2020, 9, 1), datetime(2021, 6, 10), [[xcoords[1]]], [[ycoords[1]]], cols, 3,
                      param_units, param_labels)




plt.savefig(f"plotting/main_figures/f04.jpg")
