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

# get mesh and submesh
def get_meshes(timeseries_path):
    with CheckpointFile(timeseries_path, 'r') as afile:
        mesh_ = afile.load_mesh()
        # x, y = df.SpatialCoordinate(mesh_)
        smesh_ = afile.load_mesh(name='firedrake_default_submesh')
        # sx, sy = df.SpatialCoordinate(smesh_)
    return mesh_, smesh_

# load raw data from HDF5
def load_model_output(timeseries_path):
    with h5py.File(timeseries_path, 'r') as h5file:
        us_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/Us/Us'][()].T  # (nodes, timesteps)
        m_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/m/m'][()].T
        phi_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/phi/phi'][()].T
        q_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_2/vecs/q/q'][()].T
        Q_raw = h5file['topologies/firedrake_default_submesh_topology/dms/firedrake_dm_0_1_False_1/vecs/Q/Q'][()].T
        n_idx = us_raw.shape[1]
    return us_raw, m_raw, phi_raw, q_raw, Q_raw, n_idx

# load topography
def load_topography(mesh_, sig=5):
    r_bed = gu.Raster(f"Greenland_data/BedMachineGreenland-v5_bed_smooth_sig{sig}.nc")
    r_thk = gu.Raster(f"Greenland_data/BedMachineGreenland-v5_thickness_smooth_sig{sig}.nc")
    # interpolate onto mesh
    V = df.FunctionSpace(mesh_, "CG", 1)
    meshx, meshy = zip(*hlp.get_coordinates(mesh_, "CG", 1))
    B = df.Function(V)
    H = df.Function(V)
    S = df.Function(V)
    B.dat.data[:] = r_bed.interp_points((meshx, meshy), as_array=True)
    H.dat.data[:] = r_thk.interp_points((meshx, meshy), as_array=True)
    S.interpolate(B+H)
    return B, H, S

# load velocity observations
def load_vel_obs(vel_dir, files, mesh_, element="DG", order=0, xrange=(datetime(2018,1,2),datetime(2021,12,30))):
    # extract datetime object from filenames and sort after date
    date_pattern = r'\d{2}[A-Z][a-z]{2}\d{2}'
    format_string = "%d%b%y"
    date_list = []
    for f in files:
        date_str    = re.findall(date_pattern, f)
        d0 = datetime.strptime(date_str[0], format_string)
        d1 = datetime.strptime(date_str[1], format_string)
        date_object = d0 + (d1-d0)/2
        date_list.append(date_object)
    pairs = zip(date_list, files)
    sorted_dates, sorted_files = zip(*sorted(pairs))
    sorted_dates, sorted_files = np.array(sorted_dates), np.array(sorted_files)
    i_in_range = np.where((sorted_dates>xrange[0]) & (sorted_dates<xrange[-1]))
    sorted_dates = sorted_dates[i_in_range]
    sorted_files = sorted_files[i_in_range]
    # get function space
    V_DG0 = df.FunctionSpace(mesh_, element, order)
    meshx_DG0, meshy_DG0 = zip(*hlp.get_coordinates(mesh_, element, order))
    # read files in order
    n_months = len(sorted_files)
    U_obs = []
    U_mask = []
    rasters = []
    for (i,f) in enumerate(sorted_files[:n_months]):
        r = gu.Raster(f"{vel_dir}{f}")
        delta = r.res[0]*2
        r.crop([min(meshx_DG0)-delta, min(meshy_DG0)-delta, max(meshx_DG0)+delta, max(meshy_DG0)+delta], inplace=True)
        U = df.Function(V_DG0)
        mask = df.Function(V_DG0)
        # Interpolate first
        u_interp = r.interp_points((meshx_DG0, meshy_DG0), as_array=True)
        # Then check which interpolated values are finite
        i_finite = np.where(np.isfinite(u_interp))
        U.dat.data[i_finite] = u_interp[i_finite]
        mask.dat.data[i_finite] = 1
        U_obs.append(U)
        U_mask.append(mask)
        rasters.append(r)
        # # save for visual check
        # df.VTKFile("s_obs.pvd").write(U_obs[0])
    return sorted_dates, U_obs, U_mask, rasters

# load flowlines
def segment_lengths(line):
    # Extract coordinates as a list
    coords = list(line.coords)
    # Calculate Euclidean distance between each pair
    dist = [0.0]
    dist.extend([Point(coords[i]).distance(Point(coords[i+1])) for i in range(len(coords)-1)])
    return np.cumsum(dist)

def get_s_functions(flowlines_path, fl, mesh_, smesh_, profile_width):
    # load geopackage file
    gdf = gpd.read_file(flowlines_path)

    # calculate distance along flow
    gdf['dist_along_flow'] = gdf.geometry.apply(segment_lengths)
    pts_flowl = [Point(c) for c in gdf.geometry[fl].coords]

    # create function that holds the s function (how far along the profile, in a projected sense)
    V_DG0 = df.FunctionSpace(mesh_, "DG", 0)
    meshx_DG0, meshy_DG0 = zip(*hlp.get_coordinates(mesh_, "DG", 0))
    s = df.Function(V_DG0)
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

    # save as pvd files for visual check
    df.VTKFile("s.pvd").write(s)
    df.VTKFile("s_sub.pvd").write(s_sub)

    return s, s_sub

def get_timestamps(mesh_, smesh_, B, H, us_raw, phi_raw, q_raw, Q_raw, m_raw, n_idx, idx=None):
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

