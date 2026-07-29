import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from firedrake.pyplot import tripcolor
import pandas as pd
from cycler import cycler


run_indices = [312,455]
model_labels = ["Baseline", "Reduced \nsheet flow"]
xstart,xend = datetime(2020,10,14),datetime(2021,4,16)

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
cols    = ["orchid", "forestgreen"]

vm = 6

plt.rcParams['font.size'] = 27
annotate_fs = 28
marker_size = 100

# paths
timeseries_path = f"parameter_runs/run_{run_indices[0]}/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"



def panel_letter_annotation(ax, ii):
    ax.annotate(
                f"{idx_to_letter(ii)}",
                xy=(0.03, 1.0),
                xycoords="axes fraction",
                fontsize=annotate_fs * 0.9,
                fontweight="bold",
            )

def get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, start_year=2014):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, _, _, _, _, _, n_idx, _ = load_model_output(timeseries_path)

    dates_model, i_model = get_model_dates(n_idx, xstart, xend, start_year=start_year)
    dates_model_pd = pd.to_datetime(dates_model)

    U_model = np.zeros((us_raw.shape[0], len(i_model)))
    for n, i_m in enumerate(i_model):
        U_model[:, n] = us_raw[:, i_m]

    df_monthly = pd.DataFrame(U_model.T, index=dates_model_pd)
    monthly_means = df_monthly.resample("MS").mean()

    U_diff = B
    U_diff.dat.data[:] = monthly_means.diff(axis=0).median(axis=0)
    return U_diff


def plot_model_panel(ax, fig, gs_top, ii, mesh_, run_index, model_label, B, gdf, vm, annotate_fs, xstart, xend, xcoords, ycoords, cols):
    U_diff = get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, start_year=2014)
    cl = tripcolor(U_diff, axes=ax, vmin=-vm, vmax=vm, cmap="BrBG")
    ax.scatter(xcoords, ycoords, marker_size, color="black", marker="x")
    ax.annotate("A", xy=(xcoords[0]+1e4,ycoords[0]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
    ax.annotate("B", xy=(xcoords[1]+1e4,ycoords[1]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")

    if ii == 1:
        cax = fig.add_subplot(gs_top[0, 3])
        fig.colorbar(cl, ax=cax, label="Monthly speed \nincrease " + r"($\mathrm{m\,a^{-1}}$)")
        remove_axes(cax)

    remove_axes(ax)
    ax.set_title(model_label, fontsize=annotate_fs, fontweight="bold")
    panel_letter_annotation(ax, ii+1)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="Domain")

def plot_sensitivity_panel(ax, mesh_, run_indices, xstart, xend, xc, yc, obs_files, vel_dir, colors, ii):
    dUs = []
    k_s = []
    m_basal = []

    for run_idx in run_indices:
        dU = get_mean_winter_speedup(mesh_, run_idx, xstart, xend, xc, yc, start_year=2014)
        dUs.append(dU)
        float_params = hlp.get_params_from_input_file(run_idx)
        k_s.append(float_params["k_s"])
        m_basal.append(float_params["m_basal"])
        print(f"run {run_idx}, k_s = {float_params['k_s']}")

    k_s = np.array(k_s)
    dUs = np.array(dUs)

    Uobs_glaciers = load_obs_timeseries(vel_dir, obs_files, xc, yc)[:, 0]
    obs_m = np.mean(np.diff(Uobs_glaciers))

    ax.set_prop_cycle(cycler(color=colors))
    for m in np.unique(m_basal):
        i_m = np.where(m_basal == m)
        idx = np.argsort(k_s[i_m])
        ax.plot(k_s[i_m][idx], dUs[i_m][idx], label=f"{m}", marker="o")

    ax.hlines(obs_m, np.min(k_s), np.max(k_s), color="black", ls="dashed")
    ax.legend(title=r"$m_\mathrm{basal}$ (m/yr)")
    ax.set_xlabel(r"$k_s$")
    ax.set_ylabel(r"$\Delta u_s$ (%)")
    ax.set_xscale("log")
    panel_letter_annotation(ax, ii+3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_timeseries_panel(ax, mesh_, run_indices, xstart_plot, xend_plot, xc, yc, vel_dir, vel_files, cols, ii):
    obs_dates, obs_files = get_obs_files(files, xrange=(xstart_plot, xend_plot))
    for (run_idx,c) in zip(run_indices,cols):
        float_params = hlp.get_params_from_input_file(run_idx)
        ks = float_params["k_s"]
        m  = float_params["m_basal"]
        dates_model, U_model = get_model_timeseries_for_locations(
            mesh_, run_idx, xstart_plot, xend_plot, xc, yc, start_year=2014
        )
        ax.plot(dates_model, U_model - np.mean(U_model), color=c, label=rf"$k_s = {ks}$") #+"\n"+rf"$m_\mathrm{{basal}}= {m}$")

    Uobs_glaciers = load_obs_timeseries(vel_dir, obs_files, xc, yc)[:, 0]
    ax.plot(obs_dates, Uobs_glaciers - np.nanmean(Uobs_glaciers), color="black", ls="dashed", marker="o", label="Observed")
    ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], xstart, xend, color="grey", alpha=0.5)
    ax.set_xlabel("Month of winter 2020/21")
    ax.set_ylabel(r"Speed rel. to mean ($\mathrm{m\,a^{-1}}$)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    panel_letter_annotation(ax, ii+3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

gdf = gpd.read_file(outline_path)

# load model file meshes
mesh_, smesh_ = get_meshes(timeseries_path)


# load velocity observations
files   = os.listdir(vel_dir)
obs_dates, obs_files = get_obs_files(files, xrange=(xstart,xend))
# Uobs_glaciers = load_obs_timeseries(vel_dir, obs_files, x_points, y_points)
# i_obs = np.where( (np.array(sorted_dates) > xstart) &  (np.array(sorted_dates) < xend) )[0]

raster0 = get_raster(vel_dir+obs_files[0])
nx, ny = raster0.data.shape
Uobs_time = np.zeros((raster0.data.size, len(obs_files)))
for (n,f) in enumerate(obs_files):
    raster = get_raster(vel_dir+f)
    Uobs_time[:,n] = raster.data.ravel()
    # i_nan = np.where(~np.isfinite(Uobs_time[:,n] ))[0]
    # print(i_nan)
    # if len(i_nan) > 0:
        # Uobs_time[i_nan,n] = np.nan

winter_mean = np.mean(Uobs_time,axis=1)

b    = np.nanmedian(np.diff(Uobs_time,axis=1),axis=1) #/ winter_mean
map  = b.reshape(nx,ny)
x, y = raster0.coords(grid=False)

# mask out values outside of domain
domain_outline = gu.Vector(outline_path)
mask_noglacier   = ~domain_outline.create_mask(raster0)
b[mask_noglacier.data.ravel()] = np.nan

# plot
# fig, (ax,ax2,ax3) = plt.subplots(1, 3, figsize=(21,8))

fig = plt.figure(figsize=(30, 13))
gs  = fig.add_gridspec(4, 1, left=0.01, right=0.99, top=0.92, bottom=0.1, height_ratios=[1,0.01,0.12,1])

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

for ii, (run_index, ax_model, model_label) in enumerate(zip(run_indices, [ax2, ax3], model_labels)):
    plot_model_panel(ax_model, fig, gs_top, ii, mesh_, run_index, model_label, B, gdf, vm, annotate_fs, xstart, xend, xcoords, ycoords, cols)


######################
# Sensitivity to k_s #
######################

colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

flowlines_dir = "Greenland_data/profiles/"
gl_name = "Isorlersuup"
delta = 500
d = 15e3
run_indices = [312, 452, 453, 454]
run_indices_timeseries = [312, 452]


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

obs_dates_profile, obs_files_profile = get_obs_files(files, xrange=(xstart, xend))
# obs_dates_extended, obs_files_extended = get_obs_files(files, xrange=(datetime(2020, 9, 1), datetime(2021, 6, 10)))

gs_bot = gs[3].subgridspec(1, 8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])



ax = fig.add_subplot(gs_bot[0, 1])
plot_sensitivity_panel(
    ax,
    mesh_,
    run_indices,
    xstart,
    xend,
    [[xcoords[0]]],
    [[ycoords[0]]],
    obs_files_profile,
    vel_dir,
    colors,
    0
)

ax = fig.add_subplot(gs_bot[0, 3])
plot_timeseries_panel(
    ax,
    mesh_,
    run_indices_timeseries,
    datetime(2020, 9, 1),
    datetime(2021, 6, 10),
    [[xcoords[0]]],
    [[ycoords[0]]],
    vel_dir,
    files,
    cols,
    1
)

ax = fig.add_subplot(gs_bot[0, 5])
plot_sensitivity_panel(
    ax,
    mesh_,
    run_indices,
    xstart,
    xend,
    [[xcoords[1]]],
    [[ycoords[1]]],
    obs_files_profile,
    vel_dir,
    colors,
    2
)

ax = fig.add_subplot(gs_bot[0, 7])
plot_timeseries_panel(
    ax,
    mesh_,
    run_indices_timeseries,
    datetime(2020, 8, 10),
    datetime(2021, 7, 16),
    [[xcoords[1]]],
    [[ycoords[1]]],
    vel_dir,
    files,
    cols,
    3
)

plt.savefig(f"plotting/main_figures/f05.jpg")
plt.savefig(f"plotting/output_heatmap/winter_speedup_run{run_indices[0]}_{run_indices[1]}.jpg")

