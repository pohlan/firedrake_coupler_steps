import firedrake as df
from firedrake.checkpointing import CheckpointFile
import models_main.helpers as hlp
import numpy as np
import geoutils as gu
import geopandas as gpd
from shapely.geometry import Point
import os
from datetime import datetime, timedelta
import re
import h5py

run_index = 126
timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
flowlines_path = "Greenland_data/russel/flowlines.gpkg"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
files   = os.listdir(vel_dir)

profile_width = 3e3

ds = 2.5e3
gl = 5
fl = [0,4,1,2,3][gl-1] # flow linestrings don't have the same numbering as gl; also gl starts at 1

ss = [10e3,30e3] #,43e3]  # km away from terminus at which to plot the time series

xstart,xend = datetime(2019,1,2),datetime(2022,12,30)

# load model file meshes
with CheckpointFile(timeseries_path, 'r') as afile:
        mesh_ = afile.load_mesh()
        x, y = df.SpatialCoordinate(mesh_)
        smesh_ = afile.load_mesh(name='firedrake_default_submesh')
        sx, sy = df.SpatialCoordinate(smesh_)

# Load all raw data from HDF5 at once (avoids repeated file opens)
with h5py.File(timeseries_path, 'r') as h5file:
    # Load raw datasets
    us_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/Us/Us'][()].T  # (nodes, timesteps)
    m_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/m/m'][()].T
    phi_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/phi/phi'][()].T
    q_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_2/vecs/q/q'][()].T
    Q_raw = h5file['topologies/firedrake_default_submesh_topology/dms/firedrake_dm_0_1_False_1/vecs/Q/Q'][()].T

# Set n_idx from actual transposed data shape
n_idx = us_raw.shape[1]

# FunctionSpaces and mesh coordinates for mesh_
# DGO
V_DGO = df.FunctionSpace(mesh_, "DG", 0)
meshx_DG0, meshy_DG0 = zip(*hlp.get_coordinates(mesh_, "DG", 0))
# CG1
V = df.FunctionSpace(mesh_, "CG", 1)
meshx, meshy = zip(*hlp.get_coordinates(mesh_, "CG", 1))
V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
# X = df.assemble(df.interpolate(mesh_.coordinates,V_vec))
# meshx = X.dat.data_ro[:,0]
# meshy = X.dat.data_ro[:,1]

# load topography
sig = 5
r_bed = gu.Raster(f"Greenland_data/BedMachineGreenland-v5_bed_smooth_sig{sig}.nc")
r_thk = gu.Raster(f"Greenland_data/BedMachineGreenland-v5_thickness_smooth_sig{sig}.nc")
# interpolate onto mesh
B = df.Function(V)
H = df.Function(V)
S = df.Function(V)
B.dat.data[:] = r_bed.interp_points((meshx, meshy), as_array=True)
H.dat.data[:] = r_thk.interp_points((meshx, meshy), as_array=True)
S.interpolate(B+H)

# load velocity observations
# extract datetime object from filenames and sort after date
date_pattern = r'\d{2}[A-Z][a-z]{2}\d{2}'
format_string = "%d%b%y"
date_list = []
for f in files:
    date_str    = re.search(date_pattern, f).group(0)
    date_object = datetime.strptime(date_str, format_string)
    date_list.append(date_object)
pairs = zip(date_list, files)
sorted_dates, sorted_files = zip(*sorted(pairs))
# read files in order
n_months = len(files)
U_obs = []
U_mask = []
for (i,f) in enumerate(sorted_files[:n_months]):
    r = gu.Raster(f"{vel_dir}{f}")
    delta = r.res[0]*2
    r.crop([min(meshx_DG0)-delta, min(meshy_DG0)-delta, max(meshx_DG0)+delta, max(meshy_DG0)+delta], inplace=True)
    U = df.Function(V_DGO)
    mask = df.Function(V_DGO)
    # Interpolate first
    u_interp = r.interp_points((meshx_DG0, meshy_DG0), as_array=True)
    # Then check which interpolated values are finite
    i_finite = np.where(np.isfinite(u_interp))
    U.dat.data[i_finite] = u_interp[i_finite]
    mask.dat.data[i_finite] = 1
    U_obs.append(U)
    U_mask.append(mask)

# load flowlines
def segment_lengths(line):
    # Extract coordinates as a list
    coords = list(line.coords)
    # Calculate Euclidean distance between each pair
    dist = [0.0]
    dist.extend([Point(coords[i]).distance(Point(coords[i+1])) for i in range(len(coords)-1)])
    return np.cumsum(dist)

gdf = gpd.read_file(flowlines_path)
gdf['dist_along_flow'] = gdf.geometry.apply(segment_lengths)

pts_flowl = [Point(c) for c in gdf.geometry[fl].coords]

# create function that holds the s function (how far along the profile, in a projected sense)
s = df.Function(V_DGO)
for (i,(mx, my)) in enumerate(zip(meshx_DG0, meshy_DG0)):
    dist_to_flowl_pts = Point(mx,my).distance(pts_flowl)
    i_flowl = np.argmin(dist_to_flowl_pts)
    if dist_to_flowl_pts[i_flowl] < profile_width:
        s.dat.data[i] = gdf["dist_along_flow"][fl][i_flowl]

# same for submesh
V_DGO_sub = df.FunctionSpace(smesh_, "DG", 0)
meshx_DG0_sub, meshy_DG0_sub = zip(*hlp.get_coordinates(smesh_, "DG", 0))
s_sub = df.Function(V_DGO_sub)
for (i,(mx, my)) in enumerate(zip(meshx_DG0_sub, meshy_DG0_sub)):
    dist_to_flowl_pts = Point(mx,my).distance(pts_flowl)
    i_flowl = np.argmin(dist_to_flowl_pts)
    if dist_to_flowl_pts[i_flowl] < profile_width:
        s_sub.dat.data[i] = gdf["dist_along_flow"][fl][i_flowl]


df.VTKFile("s_obs.pvd").write(U_obs[0])
df.VTKFile("s.pvd").write(s)
df.VTKFile("s_sub.pvd").write(s_sub)

def slice(s,s0,s1):
    return df.conditional(df.And(s > s0, s<=s1), 1.0, 0.0)

# load model files
def get_timestamp(idx=None):
    """Convert pre-loaded raw HDF5 data to Firedrake Functions"""
    V = df.FunctionSpace(mesh_, "CG", 1)
    V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
    V_sub = df.FunctionSpace(smesh_, "DG", 0)

    if idx is not None:
        # Return single timestep
        # Us: already scalar, just reshape
        Us = df.Function(V)
        Us.dat.data[:] = us_raw[:, idx]

        # phi and derived quantities
        phi = df.Function(V)
        phi.dat.data[:] = phi_raw[:, idx]
        # N   = df.Function(V)
        # N.interpolate(910*9.81*H-(phi-1000*9.81*B))
        pw_pi = df.Function(V)
        pw_pi.interpolate((phi-1000*9.81*B)/(910*9.81*H))

        # q: stored as 2-component vector, compute norm
        q_vec_raw = q_raw[:, idx::2]  # Extract 2 components for this timestep
        q = df.Function(V)
        q.dat.data[:] = np.sqrt(q_vec_raw[:, 0]**2 + q_vec_raw[:, 1]**2)

        # Q: scalar on submesh
        Q = df.Function(V_sub)
        Q.dat.data[:] = Q_raw[:, idx]

        # m: scalar on main mesh
        m = df.Function(V)
        m.dat.data[:] = m_raw[:, idx]

        return q, Q, Us, m
    else:
        # Return all timesteps as lists
        q_list = []
        Q_list = []
        Us_list = []
        pw_pi_list = []
        m_list = []

        for i in range(n_idx):
            # Us
            Us = df.Function(V)
            Us.dat.data[:] = us_raw[:, i]
            Us_list.append(Us)

            # phi and derived quantities
            phi = df.Function(V)
            phi.dat.data[:] = phi_raw[:, i]
            pw_pi = df.Function(V)
            pw_pi.interpolate((phi-1000*9.81*B)/(910*9.81*H))
            pw_pi_list.append(pw_pi)

            # q: compute norm from 2-component vector
            q = df.Function(V)
            # q is stored as interleaved [q_x[0], q_y[0], q_x[1], q_y[1], ...]
            q_vec_x = q_raw[::2, i]
            q_vec_y = q_raw[1::2, i]
            q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2)
            q_list.append(q)

            # Q
            Q = df.Function(V_sub)
            Q.dat.data[:] = Q_raw[:, i]
            Q_list.append(Q)

            # m
            m = df.Function(V)
            m.dat.data[:] = m_raw[:, i]
            m_list.append(m)

        return q_list, Q_list, Us_list, pw_pi_list, m_list

def get_flux_ratio(q,Q,s0s):
    Qi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi = slice(s, s0, s1)
        chi_s = slice(s_sub, s0, s1)
        # Integrate q over bulk domain and Q over submesh (boundary)
        Q_flux = df.assemble(df.avg(Q)*chi_s*df.dx(domain=smesh_)) / df.assemble(chi_s*df.dx(domain=smesh_)) # on the submesh, dx is along traces, so 1D not 2D
        q_flux = df.assemble(q*chi*df.dx) / df.assemble(chi*df.dx)
        Qi.append(Q_flux/(Q_flux+q_flux))
    return Qi

def get_Q(Q,s0s):
    Qi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi_s = slice(s_sub, s0, s1)
        Q_avg = df.assemble(Q*chi_s*df.dx(domain=smesh_)) / df.assemble(chi_s*df.dx(domain=smesh_)) # average
        Qi.append(Q_avg)
    return Qi

def get_variable(X,s0s,mask=None):
    Xi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi = slice(s, s0, s1)
        chi2 = df.Function(V_DGO)
        if mask is None:
            chi2.interpolate(1.0)
        elif df.assemble(chi*mask*df.dx) / df.assemble(chi*df.dx) < 0.1:
            Xi.append(np.nan)
            continue
        else:
            chi2.interpolate(mask)
        # print(len(np.where(np.isfinite(X.dat.data_ro))))
        # chi2.dat.data[np.where(np.isfinite(X.dat.data_ro))] = 1.0
        X_avg = df.assemble(X*chi*chi2*df.dx) / df.assemble(chi*chi2*df.dx) # average
        Xi.append(X_avg)
    return Xi

import matplotlib.pyplot as plt

# spatial, along flowline
def idx_to_month(idx):
    # idx to decimal (hard-wires 2d timestamp)
    dec = np.round((idx*2/365)%1,decimals=2)
    if dec == 0:
        return "Jan"
    elif dec == 0.25:
        return "April"
    elif dec == 0.5:
        return "July"
    elif dec == 0.7:
        return "Sep"

s0s = np.arange(0,np.max(s.dat.data_ro),step=ds)
s0s_avg = (s0s[:-1]+s0s[1:]) / 2
d_along = (s0s_avg - s0s_avg[0] ) / 1e3
print(f"# bins along profile: {len(s0s)}")

# set colors for longitudinal profile, color=point in time
colors = plt.cm.twilight(np.linspace(0, 1, 5)) #[1:]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


def plot_vel_timeseries(s,id):
    splus = ss[id]

    # get coordinates of points to plot
    p_DG0  = np.argmin(abs(s.dat.data_ro-splus))
    xi, yi = meshx_DG0[p_DG0], meshy_DG0[p_DG0]
    p_CG1  = np.argmin(np.sqrt((xi-meshx)**2+(yi-meshy)**2))
    xi2, yi2 = meshx[p_CG1], meshy[p_CG1]

    Umod_time = []
    Uobs_time = []
    m_time    = []
    # for i in range(n_idx):
        # with CheckpointFile(timeseries_path, 'r') as afile:
            # Umod_time.append(afile.load_function(mesh_, "Us", idx=i).dat.data_ro[p_CG1])
    for (Umod,m_) in zip(Us,m):
        Umod_time.append(get_variable(Umod,[splus-ds/2,splus+ds/2])[0])
        m_time.append(get_variable(m_,[splus-ds/2,splus+ds/2])[0])
    for (Uobs,mask) in zip(U_obs, U_mask):
        # Uobs_time.append(Uobs.dat.data_ro[p_DG0])
        Uobs_time.append(get_variable(Uobs,[splus-ds/2,splus+ds/2],mask=mask)[0])
    model_mean = np.array(Umod_time)[i_model].mean()
    plt.plot(dates_model, Umod_time-model_mean, label=f"model", color=colors[id], ls="dashed", lw=lw)
    i_obs = np.where( (np.array(sorted_dates) > xstart) &  (np.array(sorted_dates) < xend) )[0]
    obs_mean = np.mean(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])
    plt.plot(sorted_dates, Uobs_time-obs_mean, label=f"observations", color="black", lw=lw)
    plt.ylabel("Surface speed (m/yr)")
    plt.xlim(xstart,xend)
    ymin = min(np.array(Umod_time)[i_model].min()-model_mean, np.min(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])-obs_mean)
    ymax = max(np.array(Umod_time)[i_model].max()-model_mean, np.max(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])-obs_mean)
    plt.ylim(ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin))
    ax = plt.gca()
    ax.set_xticklabels([])
    ymin, ymax = ax.get_ylim()
    plt.vlines([datetime(2020,1,1), datetime(2021,1,1), datetime(2022,1,1), datetime(2023,1,1)], ymin-20, ymax+20, color="black", ls="dotted", alpha=0.5)
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1,7]))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2 = ax.twinx()
    ax2.fill_between(dates_model, m_time, label="melt", color="grey", alpha=0.3)
    ax2.set_ylabel("Runoff (m/yr)")
    ax2.yaxis.label.set_color("grey")
    ax2.tick_params(axis='y', colors="grey")

# plt.figure()
# plt.subplot(2,1,1)
# for idx in [365,410,456,492]:
#     q, Q, _ = get_timestamp(idx)
#     Qi = get_flux_ratio(q,Q,s0s)
#     plt.plot(d_along,Qi, label=idx_to_month(idx))
# plt.ylabel("Ratio of efficient flow")
# plt.legend()

# plt.subplot(2,1,2)
# Si = get_variable(S,s0s)
# Bi = get_variable(B,s0s)
# plt.plot(d_along,Si, label="Surface", color="cadetblue")
# plt.plot(d_along,Bi, label="Bed", color="Black")
# plt.xlabel("Distance along profile (km)")
# plt.ylabel("Elevation (m)")
# plt.legend()
# plt.savefig(f"test_profile_gl{gl}.jpg")


# time series, at certain slice of flowline
q, Q, Us, pw_pi, m = get_timestamp()
# dates_model = np.linspace(0,365*3,n_idx)
start_date = datetime(2018, 1, 1)
dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]

# i_winter = np.where( (np.array(dates_model) > datetime(2019,11,1)) &  (np.array(dates_model) < datetime(2020,3,15)) )[0]
# U_m = np.zeros((len(meshx),len(Us)))
# for (i,uu) in enumerate(Us):
#     U_m[:,i] = uu.dat.data_ro

# grd = df.Function(V)
# grd.interpolate(df.inner(df.grad(H),q))

# dv = np.zeros(len(meshx))
# gradH = np.zeros(len(meshx))
# for i_spat in range(len(meshx)):
#     tseries = U_m[i_spat,i_winter]
#     dv[i_spat] = np.diff(tseries).mean()
#     gradH[i_spat] = grd.dat.data[i_spat]

# plt.figure()
# i_plot = np.where(dv > -0.1)
# plt.scatter(dv[i_plot],gradH[i_plot])
# plt.savefig("model_winter_grad.jpg")


# set colormap for timeseries, color=distance along profile
# colors = plt.cm.Purples(np.linspace(0, 1, 5))[1:]
# colors = plt.cm.tab20b.colors[12:]
colors = ["coral","cornflowerblue"] #,"greenyellow"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

# Increase font sizes
plt.rcParams['font.size'] = 18
lw = 2.5

plt.figure(figsize=(10,18))

# Import date formatter for cleaner x-axis
import matplotlib.dates as mdates

plt.subplot(6,1,1)
for splus in ss:
    Qi_time = []
    for (qi,Qi) in zip(q,Q):
        Qi_time.append(get_flux_ratio(qi,Qi,[splus-ds/2,splus+ds/2])[0])
    plt.plot(dates_model, Qi_time, label=f"{splus*1e-3:.0f} km", lw=lw)
ax = plt.gca()
ax.set_xticklabels([])
plt.vlines([datetime(2020,1,1), datetime(2021,1,1), datetime(2022,1,1), datetime(2023,1,1)], -0.5, 1.5, color="black", ls="dotted", alpha=0.5)
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1,7]))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xlim(xstart,xend)
plt.ylim(-0.05,1.05)
plt.ylabel("Q / (Q+q)")
plt.legend()

plt.subplot(6,1,2)
ymin, ymax = 0, 1
for splus in ss:
    pw_pi_time = []
    for pw_pi_i in pw_pi:
        pw_pi_time.append(get_variable(pw_pi_i,[splus-ds/2,splus+ds/2])[0])
    plt.plot(dates_model, pw_pi_time, label=f"{splus*1e-3:.0f} km", lw=lw)
    ymin = min(np.min(np.array(pw_pi_time)[i_model]),ymin)
    ymax = max(np.max(np.array(pw_pi_time)[i_model]),ymax)
ax = plt.gca()
ax.set_xticklabels([])
plt.vlines([datetime(2020,1,1), datetime(2021,1,1), datetime(2022,1,1), datetime(2023,1,1)], ymin-1, ymax+1, color="black", ls="dotted", alpha=0.5)
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1,7]))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xlim(xstart,xend)
plt.ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
plt.ylabel("Pw / Pi")
plt.legend()

plt.subplot(6,1,3)
ymin, ymax = 1000,1000
for (ic,splus) in enumerate(ss):
    qi_time = []
    Qi_time = []
    for (Qi,qi) in zip(Q,q):
        qi_time.append(get_variable(qi,[splus-ds/2,splus+ds/2])[0])
        Qi_time.append(get_Q(Qi,[splus-ds/2,splus+ds/2])[0])
    Re = np.array(qi_time) # np.array(qi_time)/(1.793e-6*365*24*3600)
    plt.plot(dates_model, Re, label=f"{splus*1e-3:.0f} km", lw=lw, color=colors[ic])
    plt.plot(dates_model, np.array(Qi_time), label=f"{splus*1e-3:.0f} km", lw=lw, ls="-.", color=colors[ic])
    ymin = min(np.min(Re[i_model]),ymin)
    ymax = max(np.max(Re[i_model]),ymax)
ax = plt.gca()
ax.set_xticklabels([])
plt.vlines([datetime(2020,1,1), datetime(2021,1,1), datetime(2022,1,1), datetime(2023,1,1)], ymin-200, ymax+200, color="black", ls="dotted", alpha=0.5)
plt.fill_between([xstart,xend],700,1300, color="grey", alpha=0.2)
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1,7]))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xlim(xstart,xend)
plt.yscale("log")
# plt.ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
plt.ylabel("Re")
plt.legend()

plt.subplot(6,1,4)
plot_vel_timeseries(s,0)

plt.subplot(6,1,5)
plot_vel_timeseries(s,1)

plt.subplot(6,1,6)
Si = get_variable(S,s0s)
Bi = get_variable(B,s0s)
plt.plot(d_along,Si, label="Surface", color="grey", lw=lw)
plt.plot(d_along,Bi, label="Bed", color="Black", lw=lw)
plt.vlines(np.array(ss)/1e3,-500,max(Si)*1.5,ls="dashed",colors=colors, lw=lw)
plt.ylim(-max(Si)*0.05,max(Si)*1.05)
plt.xlim(-5,60)
plt.xlabel("Distance along profile (km)")
plt.ylabel("Elevation (m)")
plt.legend(loc="center right")

plt.tight_layout()
plt.savefig(f"test_profile_time_gl{gl}_run{run_index}.jpg", dpi=150, bbox_inches='tight')
