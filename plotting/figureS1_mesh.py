import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
import numpy as np
import matplotlib.pyplot as plt
import firedrake as df
from firedrake.pyplot import tripcolor, triplot
import fiona

fontsize = 29

hydro_file = f"Greenland_data/russel/hydrological_outlets_russel.gpkg"
hydro_points = fiona.open(hydro_file, mode="r")
hydro_x = np.array([i['geometry']['coordinates'][0] for i in list(hydro_points.values())])
hydro_y = np.array([i['geometry']['coordinates'][1] for i in list(hydro_points.values())])
hydro_points.close()

# fig = plt.figure(figsize=(15, 10))

fig, ax2 = plt.subplots(1,1,figsize=(17, 8))

#########
# Mesh  #
#########

mesh = df.Mesh("Greenland_data/russel/russel.msh")
colors = triplot(mesh, axes=ax2)
ax2.set_aspect('equal')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
ax2.set_xlim(-2.4e5,-2e4)
ax2.set_ylim(-2.585e6,-2.47e6)

######################
# Scalebar & N arrow #
######################

# scale bar
line_x = [-1.65e5,-1.55e5]
line_y = [-2.583e6,-2.583e6]
ax2.plot(line_x, line_y, color="black", lw=3)
ax2.annotate("10 km", [line_x[0]-5e3,line_y[0]+5e3], c="black", fontsize=fontsize)
# north arrow
ax2.arrow(
    line_x[0]+3e4, line_y[0]+1e3,
    0, 1.2e4,
    head_width=2500,
    head_length=3000,
    color='black'
)
ax2.annotate("N", [line_x[0]+3.2e4,line_y[0]+5e3], c="black", fontsize=fontsize)


#############
# Inset box #
#############

inset_x0, inset_x1 = (-2.17e5,-2.01e5)
inset_y0, inset_y1 = (-2.552e6,-2.538e6)
inset_color = "slategrey"

ax2.plot(
    [inset_x0, inset_x1, inset_x1, inset_x0, inset_x0],
    [inset_y0, inset_y0, inset_y1, inset_y1, inset_y0],
    color=inset_color
)

###################
# Inset with zoom #
###################

axins = fig.add_axes([0.01, 0.05, 0.17, 0.19])  # [left, bottom, width, height] in figure coords
inset = triplot(mesh, axes=axins)
# axins.set_ylim(-2.509e6,-2.489e6)
# axins.set_xlim(-2.335e5,-2.17e5)
# axins.set_ylim(-2.497e6,-2.489e6)
# axins.set_xlim(-2.335e5,-2.17e5)
axins.set_xlim(inset_x0, inset_x1)
axins.set_ylim(inset_y0, inset_y1)
axins.set_aspect('equal')
axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_edgecolor(inset_color)
    spine.set_linewidth(2)

# hydrological outlet
axins.scatter(hydro_x,hydro_y, 60, color="red", zorder=2)

##################
# Inset scalebar #
##################

line_x = [inset_x0+1.5e3, inset_x0+3.5e3]
line_y = [inset_y1-4e3, inset_y1-4e3]
axins.plot(line_x, line_y, color="black", lw=2.5)
axins.annotate("2 km", [line_x[0]-8e2,line_y[0]+1.1e3], c="black", fontsize=fontsize*0.75)


###################################
# Lines connecting inset with box #
###################################

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(
    ax2,
    axins,
    loc1=1,   # upper left corner of inset
    loc2=4,   # lower left corner of inset
    fc="none",
    ec=inset_color,
    lw=1,
    ls="dashed"
)

################
# Panel label  #
################
# ax2.annotate("b)", [-2.4e5,-2.47e6-1e4], xytext=[-2.4e5-2e4,-2.47e6-1e4], fontsize=22) #, fontweight="bold")




ax2.scatter(hydro_x,hydro_y, 60, color="red", zorder=2) #, edgecolors="white")

plt.tight_layout()
plt.savefig("plotting/main_figures/fS01.jpg")
