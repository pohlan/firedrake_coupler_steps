import firedrake as df
from firedrake.checkpointing import CheckpointFile
from firedrake.pyplot import tripcolor, triplot
import xarray as xr
import scipy.interpolate as itp
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import argparse
import pandas as pd
import geoutils as gu
import rasterio.transform as rt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-options_left', type=int, default=0, help='turn off annoying message')
    parser.add_argument('--run_index',dest='run_index', type=int,default=99999)
    parser.add_argument('--t_end', type=float, default=5, help='how many years of simulation')
    parser.add_argument('--k_s', type=float, default=0.05, help='sheet conductivity')
    parser.add_argument('--k_c', type=float, default=0.5, help='channel conductivity')
    parser.add_argument('--h_r', type=float, default=0.5, help='bump height')
    parser.add_argument('--l_r', type=float, default=5, help='bump spacing')
    parser.add_argument('--transition', action='store_true', help='laminar turbulent transition')
    parser.add_argument('--omega',type=float, default=1/2000)
    parser.add_argument('--alpha_s', type=float, default=1.25)
    parser.add_argument('--beta_s', type=float, default=1.5)
    parser.add_argument('--As_factor', type=float, default=2, help='A*As_factor/n^n')
    parser.add_argument('--l_c', type=float, default=10, help='sheet width below channel')
    parser.add_argument('--e_v', type=float, default=1e-4, help='englacial void ratio')
    parser.add_argument('--beta2', type=float, default=1e6, help='basal traction')
    parser.add_argument('--p', type=float, default=1, help='pressure exponent')
    parser.add_argument('--q', type=float, default=1, help='sliding exponent')
    parser.add_argument('--data_directory', default='Greenland_data/')
    parser.add_argument('--results_directory', default='parameter_runs/')
    parser.add_argument('--sig_topo', type=int, default=5, help='smoothing parameter for bed elevation and thickness; 0: no smoothing')
    parser.add_argument('--melt_input', type=str, default='MAR', help='one of `MAR` (monthly), `KAN` (daily) or `avg` (RACMO but an average, same every year)')
    parser.add_argument('--m_basal', type=float, default=0, help='Basal melt rate additional to melt_input (but without frictional heating)')
    parser.add_argument('--moulins', action='store_true', help='whether or not to route surface water through moulins; if false, use distributed melt')
    return parser.parse_args()

def save_params_to_csv(args, params_output_file, success=True):
    df_params = pd.DataFrame({"run_index":[args.run_index], "success":[success],
                              "k_s":[args.k_s], "k_c":[args.k_c], "h_r":[args.h_r], "l_r":[args.l_r], "l_c":[args.l_c], "e_v":[args.e_v],
                              "beta2":[args.beta2], "p":[args.p], "q":[args.q],
                              "transition":[args.transition], "alpha_s":[args.alpha_s], "beta_s":[args.beta_s], "omega":[args.omega], "As_factor":[args.As_factor],
                              "sig_topo":[args.sig_topo],"melt_input":[args.melt_input], "m_basal":[args.m_basal], "moulins":[args.moulins]})
    if os.path.exists(params_output_file):
        df_params.to_csv(params_output_file, mode="a", header=False, index=False)
    else:
        df_params.to_csv(params_output_file, index=False)

def get_params_from_input_file(run_index):
    input_file = f"parameter_runs/run_{run_index}/input_run{run_index}.sh"

    float_params = {}
    with open(input_file) as f:
        for line in f:
            line = line.strip()

            # skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$", line)
            if not m:
                continue

            key, value = m.groups()

            # remove surrounding quotes if present
            value = value.strip().strip("'").strip('"')

            # skip booleans
            if value in ("true", "false"):
                continue

            try:
                float_params[key] = float(value)
            except ValueError:
                # skip strings and other non-numeric values
                pass
        return float_params

def get_topography(mesh, args):
    # function spaces
    V = df.FunctionSpace(mesh, "CG", 1)
    v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
    X = df.assemble(df.interpolate(mesh.coordinates,v_dg))
    meshx = X.dat.data_ro[:,0]
    meshy = X.dat.data_ro[:,1]
    B = df.Function(V)
    H = df.Function(V)

    # load bed and thickness data, either original BedMachine or smoothed
    sig = args.sig_topo
    data_dir    = args.data_directory
    if sig==0:
        r_bed = gu.Raster(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:bed")
        r_thk = gu.Raster(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:thickness")
    else:  # higher sigma == more smoothing
        r_bed = gu.Raster(f"{data_dir}BedMachineGreenland-v5_bed_smooth_sig{sig}.nc")
        r_thk = gu.Raster(f"{data_dir}BedMachineGreenland-v5_thickness_smooth_sig{sig}.nc")

    # interpolate onto mesh
    B.dat.data[:] = r_bed.interp_points((meshx, meshy), as_array=True)
    H.dat.data[:] = r_thk.interp_points((meshx, meshy), as_array=True)
    S = B.dat.data[:] + H.dat.data[:] # surface elevation

    # set minimum ice thickness to 10
    thklim = 0
    thklim = 10
    Htemp = H.dat.data[:]
    Htemp[Htemp<thklim] = thklim
    H.dat.data[:] = Htemp

    # make bed elevation and thickness the same at bc points of individual outlets
    bc_nodes = V.boundary_nodes(1)
    for i in range(0,len(bc_nodes),2):
        nodes = bc_nodes[i:i+2]
        B.dat.data[nodes] = np.mean(B.dat.data_ro[nodes])
        H.dat.data[nodes] = np.mean(H.dat.data_ro[nodes])
    return B, H

def melt_fct_KAN(hydro, S, data_dir):
    df_KAN = pd.read_csv(data_dir+'KAN_melt.csv')
    def calc_melt(day, z):
        f_m = 0.01*365
        lapse = -0.005
        # get day
        day_floor = int(np.floor(day))
        day_ceil  = day_floor+1
        floor_weight = day_ceil - day
        ceil_weight  = day - day_floor
        i = np.where(df_KAN.dT_days == day_floor)[0][0]
        # get temperature with lapse rate
        T0 = df_KAN.T_L_smooth[i]*floor_weight + df_KAN.T_L_smooth[i+1]*ceil_weight
        z0 = df_KAN.z_L[i]*floor_weight + df_KAN.z_L[i+1]*ceil_weight
        dz = z-z0
        T = T0 + lapse*dz
        # get melt with degree-day factor
        melt = f_m*np.max([0, T])
        return melt
    def calc_m(t):
        day = t*365
        return np.array([calc_melt(day, z) for z in S])
    # first time step
    m = df.Function(hydro.V_phi)
    m.dat.data[:] = np.array([calc_melt(0.0, z) for z in S])
    return m, calc_m

def melt_fct_MAR(hydro, H, meshx, meshy, coupler):
    print("Interpolating MAR melt rates, taking a while..")
    r = gu.Raster("NETCDF:Greenland_data/MARv3.14-monthly-ERA5_1940_2023.nc:water_input_rate")
    delta = r.res[0]*2
    r.crop([min(meshx)-delta, min(meshy)-delta, max(meshx)+delta, max(meshy)+delta], inplace=True)
    year_0  = 2018 - 1940 # starts in 1940
    n_years = 6
    b_0 = year_0*12
    b_end = b_0 + n_years*12
    i_months = range(b_0,b_end+1)
    melt = np.zeros((len(H.dat.data[:]), len(i_months)))  # will interpolate onto same mesh as H
    for (n,i) in enumerate(i_months):
          melt[:,n] = r.interp_points((meshx, meshy), band=i, as_array=True) / coupler.rho_w * 12
    def calc_m(t):
        month = t*12
        month_floor = int(np.floor(month))
        month_ceil = int(np.ceil(month))
        floor_weight = month_ceil - month
        ceil_weight = month - month_floor
        return melt[:,int(month_floor)]*floor_weight + melt[:,int(month_ceil)]*ceil_weight
    # first time step
    m = df.Function(hydro.V_phi)
    m.dat.data[:] = melt[:,0]
    return m, calc_m

def get_MAR_Raster_with_transform(file):
    ds = xr.open_dataset(
        file,
        decode_times=False
    )
    da = ds.RU.sortby("y", ascending=False)  # ensures north-up
    x = da.x.values
    y = da.y.values

    dx = x[1] - x[0]
    dy = abs(y[1] - y[0])   # will typically be positive after sortby
    # transform, isn't correct when just using gu.Raster(..)
    transform = rt.from_origin(
        x.min() - dx / 2,
        y.max() + dy / 2,
        dx,
        dy
    )
    arr = da.values  # shape: (time, y, x)
    r = gu.Raster.from_array(
        arr,
        transform=transform,
        crs="EPSG:3413"
    )
    return r

def melt_fct_MAR_monthly_files(hydro, H, meshx, meshy, coupler):
    print("Interpolating MAR melt rates, taking a while..")
    melt_dir = "Greenland_data/forcing/MAR/"
    files   = os.listdir(melt_dir)
    # extract year from filename and sort in chronological order
    yr_pattern = r'\d{4}'
    format_string = "%Y"
    yrs_list = []
    for f in files:
        date_str = re.findall(yr_pattern, f)
        # d = datetime.strptime(date_str[0], format_string)
        yrs_list.append(int(date_str[0]))

    pairs = zip(yrs_list, files)
    sorted_years, sorted_files = zip(*sorted(pairs))
    # load data
    n_months = len(files)*12
    melt = np.zeros((len(H.dat.data[:]), n_months))
    for (i_file,f) in enumerate(sorted_files):
        r = get_MAR_Raster_with_transform(melt_dir+f)
        r.crop([-2.4e5, -2.585e6, 0.0, -2.47e6], inplace=True)
        for m_of_year in range(12):
            n = i_file*12 + m_of_year
            melt[:,n] = r.interp_points((meshx, meshy), band=m_of_year, as_array=True) / coupler.rho_w * 12
    def calc_m(t):
        month = t*12
        month_floor = int(np.floor(month))
        month_ceil = int(np.ceil(month))
        floor_weight = month_ceil - month
        ceil_weight = month - month_floor
        return melt[:,int(month_floor)]*floor_weight + melt[:,int(month_ceil)]*ceil_weight
    # first time step
    m = df.Function(hydro.V_phi)
    m.dat.data[:] = melt[:,0]
    return m, calc_m

def melt_fct_avg(hydro, H):
    melt = np.zeros((len(H.dat.data[:]), 12))
    for i in range(12):
        fenics_smb_file = f"Greenland_data/russel/SMB_fenics/SMB_{i}.h5"
        with CheckpointFile(fenics_smb_file, 'r') as afile:
            mesh_ = afile.load_mesh()
            SMB_ = df.Function(hydro.V_phi).interpolate(afile.load_function(mesh_, "SMB"))
            melt[:,i] = SMB_.dat.data_ro
    def calc_m(t):
        month = (t%1)*12
        month_floor = int(np.floor(month))
        month_ceil = int(np.ceil(month))
        floor_weight = month_ceil - month
        ceil_weight = month - month_floor
        return melt[:,int(month_floor%12)]*floor_weight + melt[:,int(month_ceil%12)]*ceil_weight
    # first time step
    m = df.Function(hydro.V_phi)
    m.dat.data[:] = melt[:,0]
    return m, calc_m

def moulin_dirac_from_array(mesh, x_moulins, y_moulins, M_moulins):
    sources = [([xx,yy], mm) for (xx,yy,mm) in zip(x_moulins, y_moulins, M_moulins)]
    eps = df.Constant(1.5e3)  # 'width' of delta, regularization; should be slightly larger than grid resolution (hard-wired for now)
    x = df.SpatialCoordinate(mesh)
    Qm = 0
    delta_moul = 0
    for x0_val, Q_val in sources:
        x0 = df.Constant(x0_val)
        Q_in  = df.Constant(Q_val)
        r2 = df.dot(x - x0, x - x0)   # squared distance to moulin
        delta_eps = (1 / (df.pi * eps**2)) * df.exp(-r2 / eps**2)  # integrates to one with *dx
        Qm += Q_in * delta_eps
        delta_moul += df.Constant(1.0) * delta_eps
    return Qm, delta_moul

def plot_geometry(B, H, mesh):
    # bed
    fig, axes = plt.subplots()
    cl = tripcolor(B, axes=axes)
    fig.colorbar(cl)
    plt.savefig("B.jpg")
    # thickness
    fig, axes = plt.subplots()
    cl = tripcolor(H, axes=axes)
    fig.colorbar(cl)
    plt.savefig("H.jpg")
    # mesh
    fig, axes = plt.subplots()
    colors = triplot(mesh, axes=axes)
    axes.legend()
    plt.savefig("mesh.jpg")

def scatterplt_fields(subfunctions, names, E_V, mesh, results_dir, dest_id):
    for (field, name, element) in zip(subfunctions, names, E_V.sub_elements):
        # get dof coordinates
        Vmesh = df.VectorFunctionSpace(mesh, element)
        X = df.assemble(df.interpolate(mesh.coordinates,Vmesh))
        meshx = X.dat.data[:,0]
        meshy = X.dat.data[:,1]
        # scatter plot
        plt.figure()
        plt.scatter(meshx, meshy, 10, field.dat.data_ro)
        plt.colorbar()
        plt.savefig(results_dir+name+"_"+dest_id+".jpg")

def get_coordinates(msh, el_str, order):
    v = df.VectorFunctionSpace(msh, el_str, order)
    X = df.assemble(df.interpolate(msh.coordinates,v))
    return X.dat.data_ro

def make_subDG0(mesh):
    mdim = mesh.topological_dimension()
    assert mdim == 2
    # generate DGT0 function
    DGT0 = df.FunctionSpace(mesh, "DGT", 0)
    # generate the skeleton submesh, i.e. nodes and edges, but not cells
    #   reference:  https://github.com/firedrakeproject/firedrake/blob/5144bf85ee557d62d87d6844bd82d39ccdfc4389/tests/firedrake/submesh/test_submesh_interpolate.py#L195
    facet_function = df.Function(DGT0).interpolate(df.Constant(1.))
    facet_value = 999
    mesh = df.RelabeledMesh(mesh, [facet_function], [facet_value])
    subm = df.Submesh(mesh, mdim - 1, facet_value)
    # DG0 function from submesh
    subDG0 = df.FunctionSpace(subm, "DG", 0)
    assert DGT0.dim() == subDG0.dim()         # same dimension but different dof ordering
    f_subDG0 = df.Function(subDG0)
    return f_subDG0, subm

def save_DGT0(mesh, submesh, f_DGT, f_subDG0, pvd_file, time):
    # get coordinates of dofs for both DGT0 and subDG0
    crds_DGT0 = get_coordinates(mesh, "DGT", 0)
    crds_subDG0 = get_coordinates(submesh, "DG", 0)  # doesn't need to be subm apparently
    # loop through each coordinate of DGT0 and assign value to correct index of DG0 function
    for ((xi,yi),f_val) in zip(crds_DGT0,f_DGT.dat.data_ro):
        i = np.argmin(np.sqrt((xi - crds_subDG0[:,0])**2 + (yi - crds_subDG0[:,1])**2))  # due to rounding error np.where(xi==crds_subDG0[:,0] ...) doesn't work
        f_subDG0.dat.data[i] = f_val
    return f_subDG0

def diff_to_glads_matlab(hydro, file):
    ds_mw = xr.open_dataset(file)
    x_mw = ds_mw.coords1.data[0]
    y_mw = ds_mw.coords1.data[1]
    # hard-codes that phi and h have the same dofs
    V_mesh = df.VectorFunctionSpace(hydro.mesh, hydro.E_V.sub_elements[0])
    X = df.assemble(df.interpolate(hydro.mesh.coordinates, V_mesh))
    meshx = X.dat.data[:,0]
    meshy = X.dat.data[:,1]

    F_mw = df.Function(hydro.V_phi)
    F_diff = df.Function(hydro.V_phi)
    for (f_mw, F_sol) in zip([ds_mw.N, ds_mw.h], [hydro.N, hydro.U.sub(1)]):
        F_mw.dat.data[:] = itp.griddata((x_mw,y_mw), f_mw.data[0], (meshx,meshy), method="nearest")
        F_diff.interpolate(F_mw-F_sol)
        print(np.max(np.abs(F_diff.dat.data[:]/F_mw.dat.data[:])))
        # assert np.max(np.abs(F_diff.vector()[:]/F_mw.vector()[:])) < 0.05

    # also save GlaDS-matlab as pvd to visualize in paraview
    shmip_suit = os.path.basename(file)[0:2]
    F_mw.dat.data[:] = itp.griddata((x_mw,y_mw), ds_mw.N.data[0], (meshx,meshy), method="nearest") # the default method, linear, gives nan values at the edges of the domain
    df.VTKFile(shmip_suit+"_glads-matlab_phi.pvd").write(df.project(910*9.8*hydro.H - F_mw + 1000*9.8*hydro.B, hydro.V_phi, name="phi"))
    F_mw.dat.data[:] = itp.griddata((x_mw,y_mw), ds_mw.h.data[0], (meshx,meshy), method="nearest")
    df.VTKFile(shmip_suit+"_glads-matlab_h.pvd").write(df.project(F_mw, hydro.V_phi, name="h"))
