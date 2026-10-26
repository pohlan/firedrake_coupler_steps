import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import matplotlib.pyplot as plt

# run_indices_suit = range(601,641)
# run_indices_timeseries = [478, 531]
run_indices_suit = range(811,851)
run_indices_timeseries = [812,848,706]

year0 = 2020
xstart_winter, xend_winter = datetime(year0,10,1), datetime(year0+1,4,30)
xstart_summer, xend_summer = datetime(year0,5,1), datetime(year0,9,30)

xcoords = [-205482.0,-144630.0]
ycoords = [-2527653.0,-2.52e6]

vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
timeseries_path = f"parameter_runs/run_{run_indices_suit[0]}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)

param_labels = {"k_s": r"$k_s$", "h_r": r"$h_r$", "m_basal": r"$m_\mathrm{basal}$"}
param_units  = {"k_s": r"$\mathrm{m^{\frac{7}{4}}\,kg^{-\frac{1}{2}}}$", "h_r": r"$\mathrm{m}$", "m_basal": r"$\mathrm{m\,a^{-1}}$"}

#####################
# Figure parameters #
#####################

annotate_fs = 35
plt.rcParams['font.size'] = 35
colors = plt.cm.berlin(np.linspace(0.1, 0.9, 5))

fig = plt.figure(figsize=(22, 18))
gs = fig.add_gridspec(6, 2, top=0.99, bottom=0.06, left=0.16, right=0.97, hspace=0.4, wspace=0.4, width_ratios=[1,1], height_ratios=[0.18,1,0.2,1,0.2,1])


#####################
# A and B banners   #
#####################

topax = fig.add_subplot(gs[0, 0])
topax.fill_between([0,1],0,1, color="grey", alpha=0.1)
topax.set_xlim(0,1)
topax.axis("off")
topax.annotate("A", xy=[0.5,0.4], fontsize=annotate_fs*1.2, fontweight="bold", ha="center", va="center")


topax = fig.add_subplot(gs[0, 1])
topax.fill_between([0,1],0,1, color="grey", alpha=0.1)
topax.set_xlim(0,1)
topax.axis("off")
topax.annotate("B", xy=[0.5,0.4], fontsize=annotate_fs*1.2, fontweight="bold", ha="center", va="center")


#####################
# Sensitivity plots #
#####################

# point A
ax_w = fig.add_subplot(gs[1, 0])
ax_s = fig.add_subplot(gs[3, 0])
plot_sensitivity_panel(ax_w, ax_s, mesh_, vel_dir, run_indices_suit, "k_s", "m_basal", xstart_winter, xend_winter, xstart_summer, xend_summer, [[xcoords[0]]], [[ycoords[0]]], colors, 0,
                       run_indices_timeseries, param_units, param_labels, legend=False, annotate_fs=annotate_fs, ylims_w=(-5,5.4), ylims_s=(15,68))
ylims_w_A = ax_w.get_ylim()
ylims_s_A = ax_s.get_ylim()

ax_w.annotate("Winter\nspeedup rate", xy=[-0.35, 0.5], xycoords="axes fraction", rotation=90, annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
ax_s.annotate("Max. summer\nspeed anomaly", xy=[-0.35, 0.5], xycoords="axes fraction", rotation=90, annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")

# point B
ax_w = fig.add_subplot(gs[1, 1])
ax_s = fig.add_subplot(gs[3, 1])
plot_sensitivity_panel(ax_w, ax_s, mesh_, vel_dir, run_indices_suit, "k_s", "m_basal", xstart_winter, xend_winter, xstart_summer, xend_summer, [[xcoords[1]]], [[ycoords[1]]], colors, 1,
                       run_indices_timeseries, param_units, param_labels, labelloc=(0.2,0.6), legend=True, annotate_fs=annotate_fs, ylims_w=ylims_w_A, ylims_s=ylims_s_A, scatter_annotate=True)


####################
# Timeseries plots #
####################

# A
labels = ["Lowest " + r"$k_s$","Highest " + r"$k_s$","Red. sheet flow"]
ax = fig.add_subplot(gs[5, 0])
plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, labels, xstart_winter, xend_winter, xstart_summer, xend_winter, [[xcoords[0]]], [[ycoords[0]]], cols, 4,
                      param_units, param_labels, annotate_fs=annotate_fs, legend=False)
ylims_speed = ax.get_ylim()

# B
ax = fig.add_subplot(gs[5, 1])
plot_timeseries_panel(ax, mesh_, vel_dir, run_indices_timeseries, labels, xstart_winter, xend_winter, xstart_summer, xend_winter, [[xcoords[1]]], [[ycoords[1]]], cols, 5,
                      param_units, param_labels, ylims=ylims_speed, annotate_fs=annotate_fs)


# save
plt.savefig(f"plotting/main_figures/f05.jpg")
