import firedrake as df
from firedrake.checkpointing import CheckpointFile
import models_main.helpers as hlp
import numpy as np
import geoutils as gu
import pandas as pd
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
        h_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/h/h'][()].T
        N_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_1/vecs/N/N'][()].T
        q_raw = h5file['topologies/firedrake_default_topology/dms/firedrake_dm_1_0_0_False_2/vecs/q/q'][()].T
        Q_raw = h5file['topologies/firedrake_default_submesh_topology/dms/firedrake_dm_0_1_False_1/vecs/Q/Q'][()].T
        # S_raw = h5file['topologies/firedrake_default_submesh_topology/dms/firedrake_dm_0_1_False_1/vecs/S/S'][()].T
        n_idx = us_raw.shape[1]
    return us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, n_idx, N_raw

def get_model_dates(n_idx, xstart, xend, start_year=2014):
    start_date = datetime(start_year, 1, 1)
    dates_model = [start_date + timedelta(days=2 * k) for k in range(n_idx)]
    i_model = np.where((np.array(dates_model) > xstart) & (np.array(dates_model) < xend))[0]
    return np.array(dates_model)[i_model], i_model

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

def get_profile(flowlines_path, glacier_name): #, distance_points=None, delta_point=200):
    df_profile = pd.read_csv(flowlines_path+glacier_name.replace(" ", "-") +".csv")
    dists = np.zeros(len(df_profile.X))
    for (ip, (px,py)) in enumerate(zip(df_profile.X,df_profile.Y)):
        dists[ip] = np.sqrt((px-df_profile.X[0])**2 + (py-df_profile.Y[0])**2)
    return dists, np.array(df_profile.X), np.array(df_profile.Y)

def get_obs_files(files, xrange):
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
    return sorted_dates, sorted_files

def interpolate_raster_to_points(file, xc, yc, min_x=-232904, max_x=-27945, min_y=-2578142, max_y=-2473340):
    r = gu.Raster(file)
    delta = r.res[0]*2
    r.crop([min_x-delta, min_y-delta, max_x+delta, max_y+delta], inplace=True)
    vals = r.interp_points((xc, yc), as_array=True)
    return vals

def get_raster(file, min_x=-232904, max_x=-27945, min_y=-2578142, max_y=-2473340):
    r = gu.Raster(file)
    delta = r.res[0]*2
    r.crop([min_x-delta, min_y-delta, max_x+delta, max_y+delta], inplace=True)
    return r

def interpolate_meshfct_to_profile(F, xc, yc):
    points = [(x,y) for (x,y) in zip(xc,yc)]
    vals  = F(points)
    return vals

def load_obs_timeseries(sorted_files, xc, yc):
    n_obs = len(sorted_files)
    U_obs = np.zeros((n_obs, len(xc)))
    for (i_time,f) in enumerate(sorted_files):
        r = get_raster(f)
        for (i_point,(xx,yy)) in enumerate(zip(xc,yc)):
            U_obs[i_time,i_point] = np.mean(r.interp_points((xx, yy), as_array=True))
    return U_obs

def load_model_timeseries(mesh, field, xc, yc, i_model, element="CG", order=1):
    V = df.FunctionSpace(mesh, element, order)
    F_timeseries = np.zeros((len(i_model), len(xc)))
    for (i_time, i_m) in enumerate(i_model):
        F = df.Function(V)
        F.dat.data[:] = field[:, i_m]
        for (i_point,(xx, yy)) in enumerate(zip(xc,yc)):
            F_timeseries[i_time,i_point] = np.mean(interpolate_meshfct_to_profile(F, xx, yy))
    return F_timeseries

def load_model_timeseries_Pw_Pi(mesh, phi_raw, xc, yc, i_model, B, H, element="CG", order=1):
    V = df.FunctionSpace(mesh, element, order)
    F_timeseries = np.zeros((len(i_model), len(xc)))
    for (i, i_m) in enumerate(i_model):
        phi = df.Function(V)
        phi.dat.data[:] = phi_raw[:, i_m]
        pw_pi = df.Function(V)
        pw_pi.interpolate((phi-1000*9.81*B)/(910*9.81*H))
        F_timeseries[i,:] = interpolate_meshfct_to_profile(pw_pi, xc, yc)
    return F_timeseries

# for q and Q: need to average over a certain area for it to be meaningful
def slice(s,s0,s1):
    return df.conditional(df.And(s > s0, s<=s1), 1.0, 0.0)

def get_flux_ratio(q, Q, smesh_, s, s_sub, s0s):
    Qi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi = slice(s, s0, s1)
        chi_s = slice(s_sub, s0, s1)
        # Integrate q over bulk domain and Q over submesh (boundary)
        Q_flux = df.assemble(df.avg(Q)*chi_s*df.dx(domain=smesh_)) / df.assemble(chi_s*df.dx(domain=smesh_)) # on the submesh, dx is along traces, so 1D not 2D
        q_flux = df.assemble(q*chi*df.dx) / df.assemble(chi*df.dx)
        Qi.append(Q_flux/(Q_flux+q_flux))
    return Qi

def get_Q(Q, smesh_, s_sub, s0s):
    Qi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi_s = slice(s_sub, s0, s1)
        Q_avg = df.assemble(Q*chi_s*df.dx(domain=smesh_)) / df.assemble(chi_s*df.dx(domain=smesh_)) # average
        Qi.append(Q_avg)
    return Qi

def get_q(q, mesh_, s, s0s):
    qi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi = slice(s, s0, s1)
        q_avg = df.assemble(q*chi*df.dx) / (s1-s0)
        qi.append(q_avg)
    return qi

def load_model_timeseries_discharges(mesh, smesh, q_raw, Q_raw, i_model, s, s_sub, dist, delta, element="CG", order=1):
    V = df.FunctionSpace(mesh, element, order)
    V_sub = df.FunctionSpace(smesh, "DG", 0)
    q_timeseries = np.zeros(len(i_model))
    Q_timeseries = np.zeros(len(i_model))
    for (i, i_m) in enumerate(i_model):
        # q, sheet discharge
        q_vec_x = q_raw[::2, i_m]
        q_vec_y = q_raw[1::2, i_m]
        q = df.Function(V)
        q.dat.data[:] = np.sqrt(q_vec_x**2 + q_vec_y**2)
        q_timeseries[i] = get_q(q, mesh, s, [dist-delta/2, dist+delta/2])[0]

        # Q, channel discharge; scalar on submesh
        Q = df.Function(V_sub)
        Q.dat.data[:] = Q_raw[:, i_m]
        Q_timeseries[i] = get_Q(Q, smesh, s_sub, [dist-delta/2, dist+delta/2])[0]
    return q_timeseries, Q_timeseries

# load flowlines
# def segment_lengths(coords):
#     # Calculate Euclidean distance between each pair
#     dist = [0.0]
#     dist.extend([Point(coords[i]).distance(Point(coords[i+1])) for i in range(len(coords)-1)])
#     return np.cumsum(dist)

def get_s_functions(dists,  xc, yc, mesh_, smesh_, profile_width):
    # make list from x and y coordinates
    coords = [(x,y) for (x,y) in zip(xc,yc)]
    pts_flowl = [Point(c) for c in coords]

    # create function that holds the s function (how far along the profile, in a projected sense)
    V_DG0 = df.FunctionSpace(mesh_, "DG", 0)
    meshx_DG0, meshy_DG0 = zip(*hlp.get_coordinates(mesh_, "DG", 0))
    s = df.Function(V_DG0)
    for (i,(mx, my)) in enumerate(zip(meshx_DG0, meshy_DG0)):
        dist_to_flowl_pts = Point(mx,my).distance(pts_flowl)
        i_flowl = np.argmin(dist_to_flowl_pts)
        if dist_to_flowl_pts[i_flowl] < profile_width:
            s.dat.data[i] = dists[i_flowl]

    # same for submesh
    V_DGO_sub = df.FunctionSpace(smesh_, "DG", 0)
    meshx_DG0_sub, meshy_DG0_sub = zip(*hlp.get_coordinates(smesh_, "DG", 0))
    s_sub = df.Function(V_DGO_sub)
    for (i,(mx, my)) in enumerate(zip(meshx_DG0_sub, meshy_DG0_sub)):
        dist_to_flowl_pts = Point(mx,my).distance(pts_flowl)
        i_flowl = np.argmin(dist_to_flowl_pts)
        if dist_to_flowl_pts[i_flowl] < profile_width:
            s_sub.dat.data[i] = dists[i_flowl]

    # save as pvd files for visual check
    df.VTKFile("s.pvd").write(s)
    df.VTKFile("s_sub.pvd").write(s_sub)

    return s, s_sub

# load velocity observations
def load_obs_FunctionSpace(vel_dir, files, mesh_, xrange=(datetime(2016,1,2),datetime(2024,12,30))):
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
    V_DG0 = df.FunctionSpace(mesh_, "DG", 0)
    meshx_DG0, meshy_DG0 = zip(*hlp.get_coordinates(mesh_, "DG", 0))
    # read files in order
    n_months = len(sorted_files)
    U_obs = []
    U_mask = []
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
        # # save for visual check
        # df.VTKFile("s_obs.pvd").write(U_obs[0])
    return sorted_dates, U_obs, U_mask

def get_outflow_index_skeleton_mesh(mesh, smesh):

    # Coordinates of mesh vertices
    coords = mesh.coordinates.dat.data_ro

    # Cell -> vertex connectivity
    cells = mesh.coordinates.function_space().cell_node_list

    # Exterior facets
    facets = mesh.exterior_facets

    # Cell containing each boundary facet
    cell_ids = facets.facet_cell[:, 0]

    # Local facet number within that cell
    local_facet = facets.local_facet_dat.data_ro

    cell_facets = mesh.cell_to_facets.data_ro
    markers = cell_facets[cell_ids, local_facet, 1]

    # For triangles, local facet i is opposite vertex i
    edge_vertices = np.array([
        np.delete(cells[c], lf)
        for c, lf in zip(cell_ids, local_facet)
    ])

    # Coordinates of the two endpoints of every boundary edge
    edge_coords = coords[edge_vertices]

    # Select your outflow boundary (Gmsh physical group 1)
    outflow = markers == 1

    outflow_edge_coords = edge_coords[outflow]

    coords_smesh = smesh.coordinates.dat.data_ro
    x_smesh = coords_smesh[:,0]
    y_smesh = coords_smesh[:,1]
    i_coords = []
    for (p0, p1) in outflow_edge_coords:
        xx0, yy0 = p0
        xx1, yy1 = p1
        i_coords.append(np.argmin(np.sqrt((xx0-x_smesh)**2+(yy0-y_smesh)**2)))
        i_coords.append(np.argmin(np.sqrt((xx1-x_smesh)**2+(yy1-y_smesh)**2)))

    cells_smesh = smesh.coordinates.function_space().cell_node_list
    boundary_nodes = np.array(i_coords)

    connected_edges = np.where(
        np.isin(cells_smesh[:, 0], boundary_nodes) |
        np.isin(cells_smesh[:, 1], boundary_nodes)
    )[0]

    return connected_edges


def prepare_location_inputs(xc, yc):
    if np.isscalar(xc):
        return [[xc]], [[yc]]
    return xc, yc

def get_model_timeseries_for_locations(mesh_, run_index, xstart, xend, xc, yc, start_year=2014):
    timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
    us_raw, _, _, _, _, _, n_idx, _ = load_model_output(timeseries_path)

    dates_model, i_model = get_model_dates(n_idx, xstart, xend, start_year=start_year)

    xc_input, yc_input = prepare_location_inputs(xc, yc)
    U_model = load_model_timeseries(mesh_, us_raw, xc_input, yc_input, i_model)[:, 0]
    return dates_model, U_model

def get_monthly_means_model(mesh_, run_index, xstart, xend, xc, yc, start_year=2014):
    dates_model, U_model = get_model_timeseries_for_locations(mesh_, run_index, xstart, xend, xc, yc, start_year=start_year)
    dates_model_pd = pd.to_datetime(dates_model)
    df_monthly = pd.DataFrame(U_model, index=dates_model_pd)
    monthly_means = df_monthly.resample("MS").mean()
    return monthly_means.to_numpy().ravel()
