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
from scipy.optimize import curve_fit
from plotting.figure5a_winter_speedup_map import *

plt.rcParams['font.size'] = 24
annotate_fs = 28
marker_size = 115
lw_times = 2
marker_size_times = 8

# paths
timeseries_path = f"parameter_runs/run_{run_indices_timeseries[0]}/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

run_indices_suit = range(651,706)
run_indices_timeseries = [327, 477]
# run_indices_timeseries = [626, 610]
model_labels = ["Baseline", "Reduced sheet flow"]
xstart,xend = datetime(2020,10,1),datetime(2021,4,30)

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]
cols    = ["magenta", "saddlebrown"]

param_labels = {"h_r": r"$h_r$", "alpha_s": r"$\alpha_s$"}
param_units  = {"h_r": r"$\mathrm{m}$", "alpha_s": r"-"}


fig = plt.figure(figsize=(29, 7))
gs  = fig.add_gridspec(2, 1, left=0.01, right=0.99, top=0.92, bottom=0.1, height_ratios=[0.12,1])



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




gs_bot = gs[1].subgridspec(1, 8, wspace=0, width_ratios=[0.2,1,0.3,1,0.35,1,0.3,1])

run_indices_suit = range(651, 710)
run_indices_timeseries = [652, 655]

ax = fig.add_subplot(gs_bot[0, 1])
plot_sensitivity_panel(ax, mesh_, run_indices_suit, "alpha_s", "h_r", xstart, xend, [[xcoords[0]]], [[ycoords[0]]], colors, 0,
                       param_units, param_labels, xscale="linear")

ax = fig.add_subplot(gs_bot[0, 3])
plot_timeseries_panel(ax, mesh_, run_indices_timeseries, "h_r", datetime(2020, 9, 1), datetime(2021, 6, 10), [[xcoords[0]]], [[ycoords[0]]], cols, 1,
                      param_units, param_labels)

ax = fig.add_subplot(gs_bot[0, 5])
plot_sensitivity_panel(ax, mesh_, run_indices_suit, "alpha_s", "h_r", xstart, xend, [[xcoords[1]]], [[ycoords[1]]], colors, 2,
                       param_units, param_labels, xscale="linear")

ax = fig.add_subplot(gs_bot[0, 7])
plot_timeseries_panel(ax, mesh_, run_indices_timeseries, "h_r", datetime(2020, 9, 1), datetime(2021, 6, 10), [[xcoords[1]]], [[ycoords[1]]], cols, 3,
                      param_units, param_labels)



plt.savefig(f"plotting/main_figures/fSXX.jpg")


# TODO:
# - panel labels
# - space below figure
