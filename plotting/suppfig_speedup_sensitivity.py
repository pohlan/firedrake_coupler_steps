import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.size'] = 24
annotate_fs = 28
marker_size = 115
lw_times = 2
marker_size_times = 8
ylims_A = (-3,6.5)
ylims_B = (-5.5,1.0)

model_labels = ["Baseline", "Reduced sheet flow"]
xstart_winter, xend_winter = datetime(2020,10,1), datetime(2021,4,30)
xstart_plot, xend_plot = datetime(2020, 5, 1), datetime(2021, 6, 20)

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
cols_timeseries = ["magenta", "saddlebrown"]
cols_sensitiviy = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

param_labels = {"h_r": r"$h_r$", "alpha_s": r"$\alpha_s$", "e_v": r"$e_v$", "k_c": r"$k_c$", "k_s": r"$k_s$", "m_basal":r"$m_\mathrm{basal}$"}
param_units  = {"h_r": r"$\mathrm{m}$", "alpha_s": "", "e_v": "", "k_c": "", "k_s": "", "m_basal": r"$\mathrm{m\,a^{-1}}$"}


# paths
timeseries_path = f"parameter_runs/run_478/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

# get meshes
mesh_, smesh_ = get_meshes(timeseries_path)

# plot
fig = plt.figure(figsize=(29, 21))
gs  = fig.add_gridspec(4, 1, left=0.01, right=0.99, top=0.92, bottom=0.1, hspace=0.5, height_ratios=[0.12,1,1,1])

gs_mid = gs[0].subgridspec(1,8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

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


# sensitivity to k_c

# gs_bot = gs[1].subgridspec(1, 8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

# run_indices_suit = range(668,672)
# run_indices_timeseries = [669, 671]
# # run_indices_timeseries = [626, 610]

# ax = fig.add_subplot(gs_bot[0, 1])
# plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "k_c", "m_basal", xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]], cols_sensitiviy, 0,
#                        param_units, param_labels, xscale="linear", legend=False, ylims=ylims_A)

# ax = fig.add_subplot(gs_bot[0, 3])
# plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, "k_c", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[0]]], [[ycoords[0]]], cols_timeseries, 1,
#                       param_units, param_labels)

# ax = fig.add_subplot(gs_bot[0, 5])
# plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "k_c", "m_basal", xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]], cols_sensitiviy, 2,
#                        param_units, param_labels, xscale="linear", legend=False, ylims=ylims_B)

# ax = fig.add_subplot(gs_bot[0, 7])
# plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, "k_c", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[1]]], [[ycoords[1]]], cols_timeseries, 3,
#                       param_units, param_labels)



# sensitivity to alpha_s

gs_bot = gs[2].subgridspec(1, 8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

run_indices_suit = range(701,706)
run_indices_timeseries = [701, 705]

ax = fig.add_subplot(gs_bot[0, 1])
plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "alpha_s", "h_r", xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]], cols_sensitiviy, 4,
                       param_units, param_labels, xscale="linear", legend=False, ylims=ylims_A)

ax = fig.add_subplot(gs_bot[0, 3])
plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, "alpha_s", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[0]]], [[ycoords[0]]], cols_timeseries, 5,
                      param_units, param_labels)

ax = fig.add_subplot(gs_bot[0, 5])
plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "alpha_s", "h_r", xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]], cols_sensitiviy, 6,
                       param_units, param_labels, xscale="linear", legend=False, ylims=ylims_B)

ax = fig.add_subplot(gs_bot[0, 7])
plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, "alpha_s", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[1]]], [[ycoords[1]]], cols_timeseries, 7,
                      param_units, param_labels)


# sensitivity to e_v

gs_bot = gs[3].subgridspec(1, 8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

run_indices_suit = range(801,806)
run_indices_timeseries = [801, 805]

ax = fig.add_subplot(gs_bot[0, 1])
plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "e_v", "h_r", xstart_winter, xend_winter, [[xcoords[0]]], [[ycoords[0]]], cols_sensitiviy, 8,
                       param_units, param_labels, legend=False, ylims=ylims_A)

ax = fig.add_subplot(gs_bot[0, 3])
plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, "e_v", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[0]]], [[ycoords[0]]], cols_timeseries, 9,
                      param_units, param_labels)

ax = fig.add_subplot(gs_bot[0, 5])
plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices_suit, "e_v", "h_r", xstart_winter, xend_winter, [[xcoords[1]]], [[ycoords[1]]], cols_sensitiviy, 10,
                       param_units, param_labels, legend=False, ylims=ylims_B)

ax = fig.add_subplot(gs_bot[0, 7])
plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, "e_v", xstart_winter, xend_winter, xstart_plot, xend_plot, [[xcoords[1]]], [[ycoords[1]]], cols_timeseries, 11,
                      param_units, param_labels)



plt.savefig(f"plotting/main_figures/fSXX.jpg")
