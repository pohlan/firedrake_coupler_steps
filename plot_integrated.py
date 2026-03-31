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

run_index = 137
timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
flowlines_path = "Greenland_data/russel/flowlines.gpkg"
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
files   = os.listdir(vel_dir)

# ToDo: plot also time series of..
# - melt input

profile_width = 1e3

ds = 5e3
gl = 5
fl = [0,4,1,2,3][gl-1] # flow linestrings don't have the same numbering as gl; also gl starts at 1

# load model file meshes
with CheckpointFile(timeseries_path, 'r') as afile:
        mesh_ = afile.load_mesh()
        x, y = df.SpatialCoordinate(mesh_)
        smesh_ = afile.load_mesh(name='firedrake_default_submesh')
        sx, sy = df.SpatialCoordinate(smesh_)
        n_idx = len(afile.get_timestepping_history(mesh_, "phi")['index'])

V = df.FunctionSpace(mesh_, "CG", 1)
V_DGO = df.FunctionSpace(mesh_, "DG", 0)
meshx_DG0, meshy_DG0 = zip(*hlp.get_coordinates(mesh_, "DG", 0))

# load topography
sig = 5
r_bed = gu.Raster(f"Greenland_data/BedMachineGreenland-v5_bed_smooth_sig{sig}.nc")
r_thk = gu.Raster(f"Greenland_data/BedMachineGreenland-v5_thickness_smooth_sig{sig}.nc")
# interpolate onto mesh
V_vec = df.VectorFunctionSpace(mesh_, "CG", 1)
X = df.assemble(df.interpolate(mesh_.coordinates,V_vec))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]
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
    with CheckpointFile(timeseries_path, 'r') as afile:
        if not idx==None:
            q_vec = afile.load_function(mesh_, "q", idx=idx)
            q = df.project(df.sqrt(df.dot(q_vec, q_vec)), V)
            Q = afile.load_function(smesh_, "Q", idx=idx)
            Us_vec = afile.load_function(mesh_, "Us", idx=idx)
            Us = df.project(df.sqrt(df.dot(Us_vec, Us_vec)), V)
        else:
             q = []
             Q = []
             Us = []
             for i in range(n_idx):
                  q_vec = afile.load_function(mesh_, "q", idx=i)
                  q.append(df.project(df.sqrt(df.dot(q_vec, q_vec)), V))
                  Q.append(afile.load_function(smesh_, "Q", idx=i))
                  Us_vec = afile.load_function(mesh_, "Us", idx=i)
                  Us.append(df.project(df.sqrt(df.dot(Us_vec, Us_vec)), V))
    return q, Q, Us

def get_flux_ratio(q,Q,s0s):
    Qi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi = slice(s, s0, s1)
        chi_s = slice(s_sub, s0, s1)
        # Integrate q over bulk domain and Q over submesh (boundary)
        Q_flux = df.assemble(df.avg(Q)*chi_s*df.dx(domain=smesh_)) # on the submesh, dx is along traces, so 1D not 2D
        q_flux = df.assemble(q*chi*df.dx)
        Qi.append(Q_flux/(Q_flux+q_flux))
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
q, Q, Us = get_timestamp()
ss = [15e3,40e3]  # km away from terminus at which to plot the time series
# dates_model = np.linspace(0,365*3,n_idx)
start_date = datetime(2016, 1, 1)
dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]


# set colormap for timeseries, color=distance along profile
# colors = plt.cm.Purples(np.linspace(0, 1, 5))[1:]
# colors = plt.cm.tab20b.colors[12:]
colors = ["coral","cornflowerblue"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

plt.figure(figsize=(5,7))
plt.subplot(4,1,1)
for splus in ss:
    Qi_time = []
    for (qi,Qi) in zip(q,Q):
        Qi_time.append(get_flux_ratio(qi,Qi,[splus-ds/2,splus+ds/2])[0])
    plt.plot(dates_model, Qi_time, label=f"{splus*1e-3} km")
# plt.xlim(1,3)
plt.ylabel("Ratio of efficient flow")
plt.legend()

# plt.subplot(3,1,2)
# for splus in ss:
#     Us_time = []
#     for Usi in Us:
#         Us_time.append(get_variable(Usi,[splus-ds/2,splus+ds/2])[0])
#     plt.plot(dates_model, Us_time, label=f"{splus*1e-3} km")
# plt.ylim(50,140)
# plt.ylabel("Surface speed (m/yr)")
# # plt.xlim(1,3)

# plt.subplot(3,1,3)
# for splus in ss:
#     U_time = []
#     for (Ui,mask) in zip(U_obs,U_mask):
#         U_time.append(get_variable(Ui,[splus-ds/2,splus+ds/2],mask=mask)[0])
#     plt.plot(sorted_dates, U_time, label=f"{splus} km")
# plt.xlim(datetime(2016,1,1),datetime(2021,1,1))
# # plt.ylim(0,65e3)

plt.subplot(4,1,2)
splus = ss[0]
Umod_time = []
Uobs_time = []
for Umod in Us:
    Umod_time.append(get_variable(Umod,[splus-ds/2,splus+ds/2])[0])
for (Uobs,mask) in zip(U_obs, U_mask):
    Uobs_time.append(get_variable(Uobs,[splus-ds/2,splus+ds/2],mask=mask)[0])
plt.plot(dates_model, Umod_time-np.mean(Umod_time), label=f"model", color=colors[0], ls="dashed")
obs_mean = np.mean(np.array(Uobs_time)[np.where(np.isfinite(Uobs_time))[0]])
plt.plot(sorted_dates, Uobs_time-obs_mean, label=f"observations", color="black")
plt.xlim(datetime(2016,1,10),datetime(2021,1,1))
plt.ylabel("Surface speed (m/yr)")

plt.subplot(4,1,3)
splus = ss[1]
Umod_time = []
Uobs_time = []
for Umod in Us:
    Umod_time.append(get_variable(Umod,[splus-ds/2,splus+ds/2])[0])
for (Uobs,mask) in zip(U_obs, U_mask):
    Uobs_time.append(get_variable(Uobs,[splus-ds/2,splus+ds/2],mask=mask)[0])
plt.plot(dates_model, Umod_time-np.mean(Umod_time), label=f"model", color=colors[1], ls="dashed")
obs_mean = np.mean(np.array(Uobs_time)[np.where(np.isfinite(Uobs_time))[0]])
plt.plot(sorted_dates, Uobs_time-obs_mean, label=f"observations", color="black")
plt.xlim(datetime(2016,1,10),datetime(2021,1,1))
plt.ylabel("Surface speed (m/yr)")

plt.subplot(4,1,4)
Si = get_variable(S,s0s)
Bi = get_variable(B,s0s)
plt.plot(d_along,Si, label="Surface", color="grey")
plt.plot(d_along,Bi, label="Bed", color="Black")
plt.vlines(np.array(ss)/1e3,0,1200,ls="dashed",colors=colors)
plt.xlabel("Distance along profile (km)")
plt.ylabel("Elevation (m)")
plt.legend()

plt.savefig(f"test_profile_time_gl{gl}.jpg")
