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

# run_indices_maps = [542,536]
run_indices_maps = [314,706]
# hr = hlp.get_params_from_input_file(run_indices_maps[2])["h_r"]
model_labels = ["Baseline", "Red. sheet flow", "Turbulent"]
year0 = 2020
xstart_winter, xend_winter = datetime(year0,10,1), datetime(year0+1,4,30)
xstart_summer, xend_summer = datetime(year0,5,1), datetime(year0,9,30)
xstart_plot, xend_plot = datetime(year0, 4, 15), datetime(year0+1, 6, 20)

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
x0 = np.mean(xcoords)
cols = ["coral","cornflowerblue","yellowgreen","palevioletred"]

# plotting parameters
vm = 6
vmin_s, vmax_s = 0, 110
plt.rcParams['font.size'] = 35
annotate_fs = 35
marker_size = 115
colors = plt.cm.berlin(np.linspace(0.1, 0.9, 5))
cmap_summer = cmc.nuuk # cmc.davos
cmap_winter = "BrBG"

# paths
timeseries_path = f"parameter_runs/run_{run_indices_maps[0]}/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

gdf = gpd.read_file(outline_path)

# load model file meshes
mesh_, smesh_ = get_meshes(timeseries_path)
V = df.FunctionSpace(mesh_, "CG", 1)


def get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, xstart_summer, xend_summer, start_year=2014):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, _, _, _, _, _, _, n_idx, _ = load_model_output(timeseries_path)

    dates_model, i_model_winter = get_model_dates(n_idx, xstart, xend, start_year=start_year)
    dates_model_pd_winter = pd.to_datetime(dates_model)
    dates_model, i_model_summer = get_model_dates(n_idx, xstart_summer, xend_summer)
    dates_model_pd_summer = pd.to_datetime(dates_model)

    dU_winter = np.zeros(us_raw.shape[0])
    dU_summer = np.zeros(us_raw.shape[0])
    for n in range(us_raw.shape[0]):
        U_model = us_raw[n, i_model_winter]
        df_monthly = pd.DataFrame(U_model, index=dates_model_pd_winter)
        U_monthly_winter = df_monthly.resample("MS").mean().to_numpy().ravel()
        dU_winter[n] = fit_linear_slope(U_monthly_winter) #/ np.mean(U_model)

        U_model = us_raw[n, i_model_summer]
        df_monthly = pd.DataFrame(U_model, index=dates_model_pd_summer)
        U_monthly_summer = df_monthly.resample("MS").mean().to_numpy().ravel()
        dU_summer[n] = (np.max(U_monthly_summer) - np.mean(np.concatenate([U_monthly_summer,U_monthly_winter])))

    U_diff_winter = df.Function(V)
    U_diff_winter.dat.data[:] = dU_winter

    U_diff_summer = df.Function(V)
    U_diff_summer.dat.data[:] = dU_summer

    return U_diff_winter, U_diff_summer


def plot_model_panel(ax, fig, gsc, ii, mesh_, run_index, model_label, B, gdf, vm, annotate_fs, xstart, xend, xcoords, ycoords, cols, season="winter"):
    U_diff_w, U_diff_s = get_model_winter_speedup_field(mesh_, run_index, B, xstart, xend, xstart_summer, xend_summer, start_year=2014)
    if season=="winter":
        cl = tripcolor(U_diff_w, axes=ax, vmin=-vm, vmax=vm, cmap=cmap_winter)
        ax.annotate(model_label, xy=(-0.08,0.55), xycoords="axes fraction", fontsize=annotate_fs, fontweight="bold", rotation=90, ha="center", va="center", annotation_clip=False)
    else:
        cl = tripcolor(U_diff_s, axes=ax, vmin=vmin_s, vmax=vmax_s, cmap=cmap_summer)

    ax.scatter(xcoords, ycoords, marker_size, color="black", marker="x")
    ax.annotate("A", xy=(xcoords[0]+1e4,ycoords[0]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
    ax.annotate("B", xy=(xcoords[1]+1e4,ycoords[1]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")

    remove_axes(ax)
    panel_letter_annotation(ax, ii, annotate_fs)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="Domain")
    ax.vlines(x0, ax.get_ylim()[0]+2.4e4, ax.get_ylim()[1]+1e4, color="grey", lw=6, ls="dotted")
    ax.annotate("Fig. 6 \nflux gate", xy=(x0+0.5e4, ax.get_ylim()[0]-6e3), fontsize=annotate_fs, ha="center", annotation_clip=False)


param_labels = {"k_s": r"$k_s$", "h_r": r"$h_r$", "m_basal": r"$m_\mathrm{basal}$"}
param_units  = {"k_s": r"$\mathrm{m^{\frac{7}{4}}\,kg^{-\frac{1}{2}}}$", "h_r": r"$\mathrm{m}$", "m_basal": r"$\mathrm{m\,a^{-1}}$"}


fig = plt.figure(figsize=(17, 18))
gs = fig.add_gridspec(5, 2, top=0.99, bottom=0.02, left=0.07, right=0.99, hspace=0.07, wspace=0.02, height_ratios=[0.7,1,1,0.14,1])

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
    dU_obs_winter[i] = slope # / np.mean(Uobs_time_winter[i,:])
    if (all(np.isfinite(Uobs_time_summer[i,:]))) & (all(Uobs_time_summer[i,:] > 0)):
        dU_obs_summer[i] = (np.max(Uobs_time_summer[i,:]) - np.mean(np.concatenate([Uobs_time_summer[i,:], Uobs_time_winter[i,:]]))) # / np.mean(Uobs_time[i,:])
    else:
        dU_obs_summer[i] = np.nan



# save as netcdf
# raster0.data = map
# xarray_da = raster0.to_xarray()
# xarray_da.to_netcdf(f"speedup_{year0}_{year0+1}.nc")

# mask out values outside of domain
domain_outline = gu.Vector(outline_path)
mask_noglacier   = ~domain_outline.create_mask(raster0)
dU_obs_winter[mask_noglacier.data.ravel()] = np.nan
dU_obs_summer[mask_noglacier.data.ravel()] = np.nan


# observed winter speedup
ax = fig.add_subplot(gs[1,0])
map  = dU_obs_winter.reshape(nx,ny)
x, y = raster0.coords(grid=False)
im = ax.pcolormesh(x,np.flip(y), map, vmin=-vm, vmax=vm, cmap=cmap_winter)
ax.scatter(xcoords, ycoords, marker_size, color="black", marker="x")
ax.annotate("A", xy=(xcoords[0]+1e4,ycoords[0]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
ax.annotate("B", xy=(xcoords[1]+1e4,ycoords[1]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
remove_axes(ax)
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

ax.annotate("Observed", xy=(-0.08,0.55), xycoords="axes fraction", fontsize=annotate_fs, fontweight="bold", rotation=90, ha="center", va="center", annotation_clip=False)
ax.annotate(f"{idx_to_letter(0)}", xy=(0.03,1.0), xycoords="axes fraction", fontsize=annotate_fs*0.95, fontweight="bold")
name_w = "Winter speedup rate\n"
lab_w = name_w + r"$\dot{u}_\mathrm{winter}$ " + r"($\mathrm{m\,a^{-1}}$/month)"
make_custom_cax(fig, gs[0,0], im, lab_w)

# observed summer speedup
ax = fig.add_subplot(gs[1,1])
map  = dU_obs_summer.reshape(nx,ny)
map[map==0] = np.nan
im = ax.pcolormesh(x,np.flip(y), map, vmin=vmin_s, vmax=vmax_s, cmap=cmap_summer)
ax.scatter(xcoords, ycoords, marker_size, color="black", marker="x")
ax.annotate("A", xy=(xcoords[0]+1e4,ycoords[0]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
ax.annotate("B", xy=(xcoords[1]+1e4,ycoords[1]), ha="center", va="center", fontsize=annotate_fs*0.9, fontweight="bold")
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')

remove_axes(ax)
ax.annotate(f"{idx_to_letter(1)}", xy=(0.03,1.0), xycoords="axes fraction", fontsize=annotate_fs*0.95, fontweight="bold")
name_s = "Max. summer speed\nanomaly "
lab_s = name_s + r"$\Delta u_\mathrm{max}$ " + r"$\mathrm{(m\,a^{-1})}$"
make_custom_cax(fig, gs[0,1], im, lab_s)



# #################
# # MODEL         #
# #################

B, H, S = load_topography(mesh_)

for ii, (run_index, gi, model_label) in enumerate(zip(run_indices_maps, [2,4], model_labels)):
    print(ii)
    ax_model = fig.add_subplot(gs[gi,0])
    plot_model_panel(ax_model, fig, gs[0,0], ii*2+2, mesh_, run_index, model_label, B, gdf, vm, annotate_fs, xstart_winter, xend_winter, xcoords, ycoords, cols, season="winter")

for ii, (run_index, gi, model_label) in enumerate(zip(run_indices_maps, [2,4], model_labels)):
    print(ii)
    ax_model = fig.add_subplot(gs[gi,1])
    plot_model_panel(ax_model, fig, gs[0,1], ii*2+3, mesh_, run_index, model_label, B, gdf, vm, annotate_fs, xstart_winter, xend_winter, xcoords, ycoords, cols, season="summer")


plt.savefig(f"plotting/main_figures/f04.jpg")


# #####################################
# # Separate figure for sensitivities #
# #####################################

# fig = plt.figure(figsize=(22, 18))
# gs = fig.add_gridspec(6, 2, top=0.99, bottom=0.06, left=0.16, right=0.97, hspace=0.4, wspace=0.4, width_ratios=[1,1], height_ratios=[0.18,1,0.2,1,0.2,1])


# topax = fig.add_subplot(gs[0, 0])
# topax.fill_between([0,1],0,1, color="grey", alpha=0.1)
# topax.set_xlim(0,1)
# topax.axis("off")
# topax.annotate("A", xy=[0.5,0.4], fontsize=annotate_fs*1.2, fontweight="bold", ha="center", va="center")


# topax = fig.add_subplot(gs[0, 1])
# topax.fill_between([0,1],0,1, color="grey", alpha=0.1)
# topax.set_xlim(0,1)
# topax.axis("off")
# topax.annotate("B", xy=[0.5,0.4], fontsize=annotate_fs*1.2, fontweight="bold", ha="center", va="center")


# #####################
# # Sensitivity plots #
# #####################

# # run_indices_suit = range(601,641)
# # run_indices_suit = range(811,851)
# run_indices_suit = range(811,851)
# # run_indices_timeseries = [478, 531]
# run_indices_timeseries = [812,848,706] #,836]

# # point A
# ax_w = fig.add_subplot(gs[1, 0])
# ax_s = fig.add_subplot(gs[3, 0])
# plot_sensitivity_panel(ax_w, ax_s, mesh_, vel_dir, run_indices_suit, "k_s", "m_basal", xstart_winter, xend_winter, xstart_summer, xend_summer, [[xcoords[0]]], [[ycoords[0]]], colors, 0,
#                        run_indices_timeseries, param_units, param_labels, legend=False, annotate_fs=annotate_fs, ylims_w=(-5,5.4), ylims_s=(15,68))
# ylims_w_A = ax_w.get_ylim()
# ylims_s_A = ax_s.get_ylim()

# ax_w.annotate("Winter\nspeedup rate", xy=[-0.35, 0.5], xycoords="axes fraction", rotation=90, annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
# ax_s.annotate("Max. summer\nspeed anomaly", xy=[-0.35, 0.5], xycoords="axes fraction", rotation=90, annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")

# # point B
# ax_w = fig.add_subplot(gs[1, 1])
# ax_s = fig.add_subplot(gs[3, 1])
# plot_sensitivity_panel(ax_w, ax_s, mesh_, vel_dir, run_indices_suit, "k_s", "m_basal", xstart_winter, xend_winter, xstart_summer, xend_summer, [[xcoords[1]]], [[ycoords[1]]], colors, 1,
#                        run_indices_timeseries, param_units, param_labels, labelloc=(0.2,0.6), legend=True, annotate_fs=annotate_fs, ylims_w=ylims_w_A, ylims_s=ylims_s_A, scatter_annotate=True)


# # ax = fig.add_subplot(gs[8, 1])
# # plot_sensitivity_panel_summer(ax, mesh_, vel_dir, run_indices_suit, "k_s", "m_basal", xstart_summer, xend_summer, [[xcoords[0]]], [[ycoords[0]]], colors, 4,
# #                        param_units, param_labels, annotate_fs=annotate_fs)
# # ylims_A = ax.get_ylim()

# # ax = fig.add_subplot(gs[10, 1])
# # plot_sensitivity_panel_summer(ax, mesh_, vel_dir, run_indices_suit, "k_s", "m_basal", xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]], colors, 8,
# #                        param_units, param_labels, annotate_fs=annotate_fs, legend=False) #, obs_note="left")
# # ylims_B = ax.get_ylim()



# ####################
# # Timeseries plots #
# ####################

# # A
# labels = ["Lowest " + r"$k_s$","Highest " + r"$k_s$","Red. sheet flow"]
# ax = fig.add_subplot(gs[5, 0])
# plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, labels, xstart_winter, xend_winter, xstart_plot, xend_winter, [[xcoords[0]]], [[ycoords[0]]], cols, 4,
#                       param_units, param_labels, annotate_fs=annotate_fs, legend=False)
# ylims_speed = ax.get_ylim()

# # B
# ax = fig.add_subplot(gs[5, 1])
# plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, labels, xstart_winter, xend_winter, xstart_plot, xend_winter, [[xcoords[1]]], [[ycoords[1]]], cols, 5,
#                       param_units, param_labels, ylims=ylims_speed, annotate_fs=annotate_fs)


# # save
# plt.savefig(f"plotting/main_figures/f05.jpg")
