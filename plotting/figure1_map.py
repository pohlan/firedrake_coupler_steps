import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
from plotting.plotting_functions import *
from firedrake.pyplot import tripcolor, triplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from datetime import datetime, timedelta
import firedrake as df
from firedrake.pyplot import tripcolor
import cmcrameri.cm as cmc

plt.rcParams['font.size'] = 16
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]

outline_path   = "Greenland_data/russel/russel_domain.gpkg"
outline_GrIS_path = "Greenland_data/gris-outline-imbie-1980_updated_crs.shp"
flowlines_path = "Greenland_data/russel/flowlines.gpkg"


# domain outline
gdf = gpd.read_file(outline_path)


def get_topo(file):
    r = gu.Raster(file)
    # delta = r.res[0]*2
    r.crop([-2.4e5, -2.585e6, 0, -2.47e6], inplace=True)
    outline = gu.Vector(outline_path)
    mask   = ~outline.create_mask(r)
    r.set_mask(mask)
    return r

######################
# Full domain -- bed #
######################

fig = plt.figure(figsize=(15, 10))
# use 2x2 grid so that colorbar has its own axis and doesn't mess up x-axis alignment between plots
gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

r_bed = get_topo("Greenland_data/BedMachineGreenland-v5_bed_smooth_sig5.nc")
data    = r_bed.data.squeeze()
x, y    = r_bed.coords(grid=False)
im = ax1.pcolormesh(x,np.flip(y),data,cmap="terrain")
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Bed elevation (m)")

# fig.subplots_adjust(right=4)

# format
ax1.set_aspect('equal')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
ax2.set_ylim(-2.585e6,-2.47e6)
ax2.set_xlim(-2.4e5,-2e4)
# domain outline
gdf.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=1, label='Domain')


######################
# Scalebar & N arrow #
######################

# scale bar
line_x = [-2.35e5,-2.25e5]
line_y = [-2.58e6,-2.58e6]
ax1.plot(line_x, line_y, color="black", lw=3)
ax1.annotate("10 km", [-2.4e5,line_y[0]+5e3], c="black", fontsize=20)
# north arrow
ax1.arrow(
    -2.05e5, -2.58e6,
    0, 1.2e4,
    head_width=2500,
    head_length=3000,
    color='black'
)
ax1.annotate("N", [-2.02e5,line_y[0]+5e3], c="black", fontsize=20)


######################
# Surface elevation  #
######################

r_thick = get_topo("Greenland_data/BedMachineGreenland-v5_thickness_smooth_sig5.nc")
r_surf  = r_bed+r_thick
data    = r_surf.data.squeeze()
x, y    = r_surf.coords(grid=False)
cs = ax1.contour(x, np.flip(y), data,
                levels=np.arange(0, 3000, 100),
                colors="black",
                linewidths=0.5,
)
labels = ax1.clabel(cs, fmt="%d m", fontsize=12, levels=np.arange(900, 3000, 300), rightside_up=False)



######################
# Flowline profiles  #
######################

gls = [1,3,4,5]
d_upglacier = [10e3,30e3,50e3,100e3]
colors = ["coral","cornflowerblue","yellowgreen","palevioletred"]

annotate_offsets = [[-1e4,1e4],[-4e4,-5e3],[-3.5e4,-1e4],[-2.7e4,-1e4]]
gdf_flowlines = gpd.read_file(flowlines_path)

for (gl,col,strng,offset_crd) in zip(gls,colors,["Isunnguata\n Sermia","Ørkendalen","Isorlersuup","Name\nunknown"],annotate_offsets):
    fl = [0,4,1,2,3][gl-1]
    coords = list(gdf_flowlines.geometry[fl].coords)
    x = [x for (x,y) in coords]
    y = [y for (x,y) in coords]
    ax1.plot(x,y,color=col,lw=3)
    dists = segment_lengths(gdf_flowlines.geometry[fl])
    marker_x = []
    marker_y = []
    for d in d_upglacier:
        marker_x.append(coords[np.argmin(abs(dists-d))][0])
        marker_y.append(coords[np.argmin(abs(dists-d))][1])
    ax1.scatter(marker_x, marker_y, 40, c=col, edgecolors='black', marker="s",zorder=3)
    ax1.annotate(strng,[marker_x[0],marker_y[0]], xytext=[marker_x[0]+offset_crd[0],marker_y[0]+offset_crd[1]], c=col, fontsize=15)

fig.tight_layout()


################
# Panel label  #
################
ax1.annotate("a)", [-2.4e5,-2.47e6-1e4], xytext=[-2.4e5-2e4,-2.47e6-1e4], fontsize=22) #, fontweight="bold")



######################

######################
# Panel B -- Mesh    #
######################

######################

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


##################
# Inset scalebar #
##################

line_x = [inset_x0+1.5e3, inset_x0+3.5e3]
line_y = [inset_y1-4e3, inset_y1-4e3]
axins.plot(line_x, line_y, color="black", lw=2.5)
axins.annotate("2 km", [line_x[0]-8e2,line_y[0]+1.1e3], c="black", fontsize=18)


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
ax2.annotate("b)", [-2.4e5,-2.47e6-1e4], xytext=[-2.4e5-2e4,-2.47e6-1e4], fontsize=22) #, fontweight="bold")


######################

######################
# Greenland inset    #
######################

######################

gdf_GrIS = gpd.read_file(outline_GrIS_path)
print(gdf_GrIS.crs)
axins2 = fig.add_axes([0.8, 0.17, 0.17, 0.3])  # [left, bottom, width, height] in figure coords
gdf_GrIS.plot(ax=axins2, facecolor='none', edgecolor='black', linewidth=0.5)

# axins2.scatter([np.mean(x)], [np.mean(y)], 20, "red", marker="o")
gdf.plot(ax=axins2, facecolor='firebrick', alpha=0.8, edgecolor='black', linewidth=1.0)

axins2.set_aspect('equal')
axins2.set_xticks([])
axins2.set_yticks([])
axins2.spines[['right', 'left', 'top', 'bottom']].set_visible(False)

# arrow
x, y    = r_surf.coords(grid=False)
x0 = np.mean(x)
y0 = np.mean(y)
axins2.arrow(
    x0+3e5, y0+8e5,
    -1.8e5, -5.6e5,
    lw=2.5,
    head_width=100000,
    head_length=90000,
    color='firebrick'
)

################
# Panel label  #
################
ax2.annotate("c)", [-2e5,-2.47e6-1e4], xytext=[-1e4,-2.47e6-1e4], fontsize=22) #, fontweight="bold")


plt.savefig("plotting/main_figures/f01.jpg")



#############################################################################

# plot area of domain and mesh resolution...


# V = df.FunctionSpace(mesh, "DG", 0)
# X = df.Function(V).interpolate(1)
# area = df.assemble(X * df.dx(domain=mesh))
# print(area*1e-6) # in km^2


# coords = mesh.coordinates.dat.data_ro
# cells = mesh.coordinates.cell_node_map().values

# h = []
# for c in cells:
#     X = coords[c]
#     # pairwise distances in element
#     d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
#     h.append(np.max(d))

# h = np.array(h)

# print("element h_min:", h.min())
# print("element h_max:", h.max())

# # indices of 10 smallest elements
# idx = np.argsort(h)[:20]

# print("10 smallest element sizes:")
# for i in idx:
#     print(i, h[i])

# idx = np.argsort(h)
# print("10 smallest element sizes:")
# for i in idx[-20:]:
#     print(i, h[i])