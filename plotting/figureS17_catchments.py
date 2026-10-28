import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotting.plotting_functions import *

outline   = "Greenland_data/russel/russel_domain.gpkg"
outline_small   = "Greenland_data/russel/domain_smaller.gpkg"
outline_Mank   = "Greenland_data/russel/domain_Mankoff_k09_5glaciers.gpkg"

catchments_Yang = "Greenland_data/Yang_Smith_2016/ds01/catchments_LC80070132013231LGN00_B8.shp"
catchments_Mankoff = "Greenland_data/russel/basins_Mankoff_k_0.9.gpkg"

# plotting parameters
ylims=(-2.62e6,-2.47e6)
plt.rcParams['font.size'] = 30
annotate_fs = 30
lw_outlines = 3.5

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(18,8.3))
fig.subplots_adjust(left=0.01, right=0.95, bottom=0.25, top=1.0, wspace=0.05)
# Yang&Smith catchments
outline_path = outline
gdf = gpd.read_file(outline)
gdf_small = gpd.read_file(outline_small)
gg = gpd.read_file(catchments_Yang)
gg.to_crs(gdf.crs, inplace=True)
# plot outlines
gg.plot(ax=ax1, facecolor="none", edgecolor="cornflowerblue", linewidth=0.7, label="Yang and Smith (2016) catchments")
gdf.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=lw_outlines, label='Domain used for main results')
gdf_small.plot(ax=ax1, facecolor='none', edgecolor='peru', linewidth=lw_outlines, label='Smaller domain')
handles = [
    Line2D([0], [0], color="cornflowerblue", linewidth=2.5,
           label="Yang and Smith (2016) catchments"),
    Line2D([0], [0], color="black", linewidth=lw_outlines,
           label="Domain used for main results"),
    Line2D([0], [0], color="peru", linewidth=lw_outlines,
           label="Smaller domain"),
]
remove_axes(ax1, ylims=ylims)
ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), handlelength=1)
ax1.annotate(f"{idx_to_letter(0)}", xy=(0.03,0.95), xycoords="axes fraction", fontsize=annotate_fs*0.9, fontweight="bold")

# Mankoff catchments
gg_Mank = gpd.read_file(catchments_Mankoff)
gg_Mank.to_crs(gdf.crs, inplace=True)
# plot outlines
gdf_Mank = gpd.read_file(outline_Mank)
gg_Mank.plot(ax=ax2, facecolor="none", edgecolor="cornflowerblue", linewidth=0.5, label="Yang and Smith (2016) catchments")
gdf.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=2.5, label='Domain used for main results')
gdf_Mank.plot(ax=ax2, facecolor='none', edgecolor='indianred', linewidth=2.5, label='Smaller domain')
handles = [
    Line2D([0], [0], color="cornflowerblue", linewidth=1.1,
           label="Mankoff et al. (2020) catchments"),
    Line2D([0], [0], color="black", linewidth=2.5,
            label="Domain used for main results"),
    Line2D([0], [0], color="indianred", linewidth=2.5,
           label="Larger domain"),
]
remove_axes(ax2, ylims=ylims)
ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), handlelength=1)
ax2.annotate(f"{idx_to_letter(1)}", xy=(0.03, 0.95), xycoords="axes fraction", fontsize=annotate_fs*0.9, fontweight="bold", annotation_clip=False)

plt.savefig("plotting/main_figures/fS17.jpg")

