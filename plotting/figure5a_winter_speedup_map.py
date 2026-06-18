import os
import sys
# Add parent directory to path so imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.loading_functions import *
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, triplot
import pandas as pd

run_index = 181
xstart,xend = datetime(2017,11,1),datetime(2018,4,30)

# paths
timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
outline_path   = "Greenland_data/russel/russel_domain.gpkg"

gdf = gpd.read_file(outline_path)

# load model file meshes
mesh_, smesh_ = get_meshes(timeseries_path)

# load model output from hdf5 file
us_raw, m_raw, phi_raw, q_raw, Q_raw, n_idx = load_model_output(timeseries_path)

# load velocity observations
files   = os.listdir(vel_dir)
sorted_dates, U_obs, U_obs_mask, rasters = load_vel_obs(vel_dir, files, mesh_, element="CG", order=1, xrange=(xstart,xend))
n_tsteps = len(sorted_dates)
# i_obs = np.where( (np.array(sorted_dates) > xstart) &  (np.array(sorted_dates) < xend) )[0]

nx, ny = rasters[0].data.shape
Uobs_time = np.zeros((rasters[0].data.size,n_tsteps))
for n in range(n_tsteps):
    Uobs_time[:,n] = rasters[n].data.ravel()
    # i_nan = np.where(~np.isfinite(Uobs_time[:,n] ))[0]
    # print(i_nan)
    # if len(i_nan) > 0:
        # Uobs_time[i_nan,n] = np.nan

winter_mean = np.mean(Uobs_time,axis=1)

b = np.nanmedian(np.diff(Uobs_time,axis=1),axis=1) #/ winter_mean
map = b.reshape(nx,ny)
x, y    = rasters[0].coords(grid=False)

# mask out values outside of domain
domain_outline = gu.Vector(outline_path)
mask_noglacier   = ~domain_outline.create_mask(rasters[0])
b[mask_noglacier.data.ravel()] = np.nan

# plot
fig, (ax,ax2,ax3) = plt.subplots(1, 3, figsize=(13,5))
im = ax.pcolormesh(x,np.flip(y),map, vmin=-9, vmax=9, cmap="BrBG") #, cmap='viridis')
fig.colorbar(im, ax=ax, label="Speed increase per month (m/yr)")
ax.set_aspect('equal')
ax.set_xlim(-2.35e5,-1.5e5)
ax.set_ylim(-2.565e6,-2.47e6)
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='Domain')
# plt.savefig("test.jpg")


###########################################
# On firedrake function instead of raster #
# less good, one can see the triangles..  #
###########################################


# Uobs_time = np.zeros((len(U_obs[0].dat.data_ro),n_tsteps))
# for n in range(n_tsteps):
#     Uobs_time[:,n] = U_obs[n].dat.data_ro

# b_obs = np.nanmedian(np.diff(Uobs_time,axis=1),axis=1)

# B_obs = U_obs[0]
# B_obs.dat.data[:] = b_obs


# fig, axes = plt.subplots(figsize=(5,6))
# cl = tripcolor(B, axes=axes, vmin=-10, vmax=10, cmap="bwr")
# fig.colorbar(cl)
# axes.set_aspect('equal')
# axes.set_xlim(-2.4e5,-1.5e5)
# gdf.plot(ax=axes, facecolor='none', edgecolor='black', linewidth=1, label='Domain')
# plt.savefig("test.jpg")




#################
# MODEL         #
#################

# get model time series
start_date = datetime(2016, 1, 1)
dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]

B, H, S = load_topography(mesh_)
q, Q, Us, pw_pi, m = get_timestamps(mesh_, smesh_, B, H, us_raw, phi_raw, q_raw, Q_raw, m_raw, n_idx)

Umodel_time = np.zeros((len(Us[0].dat.data_ro),len(i_model))) # space, time
qmodel_time = np.zeros((len(q[0].dat.data_ro),len(i_model))) # space, time
Qmodel_time = np.zeros((len(Q[0].dat.data_ro),len(i_model))) # space, time
q_facets_time = np.zeros((len(Q[0].dat.data_ro),len(i_model))) # space, time

for n in range(len(i_model)):
    Umodel_time[:,n] = Us[i_model[n]].dat.data_ro
    qmodel_time[:,n] = q[i_model[n]].dat.data_ro
    Qmodel_time[:,n] = Q[i_model[n]].dat.data_ro
    q_facets = Q[0]
    q_facets.interpolate(q[i_model[n]])
    q_facets_time[:,n] = q_facets.dat.data_ro
    # length of facet
    V_edge = df.FunctionSpace(mesh_, "DGT", 0)
    q_facets.project(df.FacetArea(smesh_),V_edge)
    print(q_facets.dat.data_ro)

# calculate monthly averages with panda
dates = pd.to_datetime(np.array(dates_model)[i_model])
# Create DataFrame with time as index
df_monthly = pd.DataFrame(
    Umodel_time.T,  # shape (time, space)
    index=dates
)
monthly_means = df_monthly.resample('MS').mean()
winter_means  = monthly_means.mean(axis=0)

U_diff = Us[0]
U_diff.dat.data[:] = monthly_means.diff(axis=0).median(axis=0) #/ winter_means

cl = tripcolor(U_diff, axes=ax2, vmin=-9, vmax=9, cmap="BrBG")
# cl = tripcolor(q[1], axes=ax2, vmax=4e4, cmap="BrBG")
fig.colorbar(cl)
ax2.set_aspect('equal')
ax2.set_xlim(-2.35e5,-1.5e5)
ax2.set_ylim(-2.565e6,-2.47e6)
gdf.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=1, label='Domain')


#############################

df_q = pd.DataFrame(
    qmodel_time.T,  # shape (time, space)
    index=dates
)
q_vals = np.array(df_q.resample('MS').mean().diff(axis=0).median(axis=0))
df_Q = pd.DataFrame(
    Qmodel_time.T,  # shape (time, space)
    index=dates
)
Q_vals = np.array(df_Q.resample('MS').mean().diff(axis=0).mean(axis=0))
# Q_vals = np.array(df_Q.resample('MS').mean().iloc[0,:])
# print(Q_vals.shape)
df_q = pd.DataFrame(
    q_facets_time.T,  # shape (time, space)
    index=dates
)
# q_vals_facets = np.array(df_q.resample('MS').mean().diff(axis=0).median(axis=0))
q_vals_facets = np.array(df_q.resample('MS').mean().iloc[0,:])


Qq_ratio = df_Q.resample('MS').mean() / df_q.resample('MS').mean()
Qq = np.array(Qq_ratio.diff(axis=0).mean(axis=0))

ii = np.where(S.dat.data_ro > 10)
# im = ax3.scatter(B_obs.dat.data_ro[ii], U_diff.dat.data_ro[ii], 2, S.dat.data_ro[ii])
im = ax3.scatter(Qq[ii], U_diff.dat.data_ro[ii], 2, S.dat.data_ro[ii])
# im = ax3.scatter(Q_vals[ii]/(q_vals_facets[ii]+Q_vals[ii]), U_diff.dat.data_ro[ii], 2, S.dat.data_ro[ii])
fig.colorbar(im, ax=ax3, label="Surface elevation (m)")
# ax3.set_xlim(-1e6,1e6)
# ax3.set_aspect('equal')
# ax3.plot([-5,5],[-5,5],color="black")


# cl = tripcolor(q[1], axes=ax3, vmax=4e4) #, cmap="BrBG")
# fig.colorbar(cl)
# ax3.set_aspect('equal')
# ax3.set_xlim(-2.35e5,-1.5e5)
# ax3.set_ylim(-2.565e6,-2.47e6)
# gdf.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=1, label='Domain')


plt.savefig(f"plotting/output_heatmap/winter_speedup_run{run_index}.jpg")

