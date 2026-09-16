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
import matplotlib.colors as colors

xstart, xend = datetime(2020,4,1), datetime(2020,10,30)

run_indices_maps = [314, 478]
model_labels = ["Baseline", "Reduced sheet flow"]

# plotting parameters
vmax = 305
vmin = 50
plt.rcParams['font.size'] = 23
annotate_fs = 24
marker_size = 115
cmap_speed = cmc.lapaz
cmap_melt = cmc.lajolla

# paths
timeseries_path = f"parameter_runs/run_{run_indices_maps[0]}/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

# load outline
gdf = gpd.read_file(outline_path)
domain_outline = gu.Vector(outline_path)

# load model file meshes
mesh_, smesh_ = get_meshes(timeseries_path)
V = df.FunctionSpace(mesh_, "CG", 1)

def get_model_speed_melt(run_index, V, xstart, xend, dmonths=2):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, m_raw, _, _, _, _, _, n_idx, _ = load_model_output(timeseries_path)

    dates_model, i_model = get_model_dates(n_idx, xstart, xend)
    dates_model_pd = pd.to_datetime(dates_model)

    n_maps = int(np.ceil((xend.month-xstart.month+1)/dmonths))
    Us = np.zeros((us_raw.shape[0],n_maps))
    ms = np.zeros((us_raw.shape[0],n_maps))
    for n in range(us_raw.shape[0]):
        U_model = us_raw[n, i_model]
        m_model = m_raw[n, i_model]
        df_U = pd.DataFrame(U_model, index=dates_model_pd)
        df_monthly = df_U.resample("MS").mean()
        df_monthly.index = (df_monthly.index + (df_monthly.index + pd.offsets.MonthEnd(0) - df_monthly.index) / 2)
        U_monthly = df_monthly.to_numpy().ravel()
        Us[n,:] = U_monthly[::dmonths]

        df_m = pd.DataFrame(m_model, index=dates_model_pd)
        df_monthly = df_m.resample("MS").mean()
        df_monthly.index = (df_monthly.index + (df_monthly.index + pd.offsets.MonthEnd(0) - df_monthly.index) / 2)
        m_monthly = df_monthly.to_numpy().ravel()
        ms[n,:] = m_monthly[::dmonths]

    U_maps = []
    m_maps = []
    for i in range(n_maps):
        Umap = df.Function(V)
        Umap.dat.data[:] = Us[:,i]
        U_maps.append(Umap)
        mmap = df.Function(V)
        mmap.dat.data[:] = ms[:,i]
        m_maps.append(mmap)
    return U_maps, m_maps, df_monthly.index[::dmonths]

def plot_model_panel(U, ax, fig, gs, ii, iy, gdf, annotate_fs, vmin, vmax, cmap):
    if vmin < 1:
        cl = tripcolor(U, axes=ax, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        cl = tripcolor(U, axes=ax, vmin=vmin, vmax=vmax, cmap=cmap)

    if ii == 0:
        cax = fig.add_subplot(gs[1, 4])
        fig.colorbar(cl, ax=cax, label="Surface speed " + r"($\mathrm{m\,a^{-1}}$)")
        remove_axes(cax)
    elif vmin < 1:
        cax = fig.add_subplot(gs[2, 4])
        fig.colorbar(cl, ax=cax, label="Melt input " + r"($\mathrm{m\,a^{-1}}$)")
        remove_axes(cax)

    remove_axes(ax)
    panel_letter_annotation(ax, 4*iy+ii+1, annotate_fs)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="Domain")


def get_obs_speed(xstart,xend,dmonths=2):
    # load velocity observations
    files = glob.glob(vel_dir+"*vv*.tif")
    obs_dates, obs_files = get_obs_files(files, xrange=(xstart,xend))

    raster0 = get_raster(obs_files[0])
    mask_noglacier   = ~domain_outline.create_mask(raster0)

    obs_files = obs_files[::dmonths]
    U_maps = [np.zeros(raster0.data.shape) for i in range(len(obs_files))]
    for (n,fvv) in enumerate(obs_files):
        raster = get_raster(fvv)
        dat    = raster.data
        dat[mask_noglacier.data] = np.nan
        dat[np.where(dat < 0)] = np.nan
        U_maps[n][:] = raster.data
    x, y = raster0.coords(grid=False)

    return x, y, U_maps, obs_dates[::dmonths]


# observations
x, y, Uobs, obs_dates = get_obs_speed(xstart,xend)

fig = plt.figure(figsize=(30, len(Uobs)*4))
gs  = fig.add_gridspec(len(Uobs), 6, left=0.07, right=0.99, top=0.97, bottom=0.01, hspace=0, width_ratios=[1,1,1,1,0.15,0.2])

for (iy,U) in enumerate(Uobs):
    ax = fig.add_subplot(gs[iy,0])
    month = obs_dates[iy].strftime("%b")
    ax.annotate(f"{month}", xy=[-0.25,0.5], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
    im = ax.pcolormesh(x,np.flip(y), U, vmin=vmin, vmax=vmax, cmap=cmap_speed)
    remove_axes(ax)
    panel_letter_annotation(ax, 4*iy, annotate_fs)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="Domain")
    if iy == 0:
        ax.annotate("Observed", xy=[0.5,1.15], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")


# model runs
for ii, (run_index, model_label) in enumerate(zip(run_indices_maps, model_labels)):
    U_maps, m_maps, dates = get_model_speed_melt(run_index, V, xstart, xend)
    for (iy, (U, m)) in enumerate(zip(U_maps, m_maps)):
        ax = fig.add_subplot(gs[iy,ii+1])
        plot_model_panel(U, ax, fig, gs, ii, iy, gdf, annotate_fs, vmin, vmax, cmap_speed)
        if iy == 0:
            ax.annotate(model_label, xy=[0.5,1.15], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")
        if ii == 1:
            ax = fig.add_subplot(gs[iy,3])
            plot_model_panel(m, ax, fig, gs, ii+1, iy, gdf, annotate_fs, 0.02, 30, cmap_melt)
        if (iy==0) & (ii==1):
            ax.annotate("Melt forcing", xy=[0.5,1.15], xycoords="axes fraction", annotation_clip=False, fontsize=annotate_fs, fontweight="bold", ha="center", va="center")



plt.savefig(f"plotting/main_figures/fS01d.jpg")
