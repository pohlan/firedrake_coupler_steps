import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
from models_main.coupled_model import GLADS, SpecFO, Coupler
import models_main.helpers as hlp
import numpy as np
import pandas as pd

# import rasterio as rio
# from scipy.ndimage import gaussian_filter
import geoutils as gu
from firedrake.__future__ import interpolate

s_per_hour = 3600
s_per_day  = s_per_hour * 24

args = hlp.get_args()   # command line arguments

data_dir    = args.data_directory
results_dir = args.results_directory+"/run_{}/".format(args.run_index)
params_output_file = args.results_directory+"parameter_runs.csv"

# m          = 3e-13
# seasonal   = False

# mesh
mesh_file = data_dir+'russel/russel.msh'
mesh = df.Mesh(mesh_file)

# time stepping
dt_max = 20/365
dt_min = 1e-3/365
timestep_increase_fraction = 1.05
timestep_reduction_fraction = 0.5
day = 1/365
hour = day/24
def get_dt(m):
    return max(2.0*hour, 20*hour + hour*(2.0-20)/(15-1e-14) * (m-1e-14))

# geometry
V = df.FunctionSpace(mesh, "CG", 1)
v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
X = df.assemble(interpolate(mesh.coordinates,v_dg))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]
B = df.Function(V)
H = df.Function(V)

r_bed = gu.Raster(f"{data_dir}BedMachineGreenland-v5_bed_smooth.nc")
B.dat.data[:] = r_bed.interp_points((meshx, meshy))
r_thk = gu.Raster(f"{data_dir}BedMachineGreenland-v5_thickness_smooth.nc")
H.dat.data[:] = r_thk.interp_points((meshx, meshy))

# set minimum ice thickness to 10
thklim = 0
thklim = 10
Htemp = H.vector().get_local()
Htemp[Htemp<thklim] = thklim
H.vector().set_local(Htemp)

# make bed elevation and thickness the same at bc points of individual outlets
# bc_nodes = V.boundary_nodes(1)
# for i in range(0,len(bc_nodes),2):
#     nodes = bc_nodes[i:i+2]
#     B.vector()[nodes] = np.mean(B.vector()[nodes])
#     H.vector()[nodes] = np.mean(H.vector()[nodes])

# initiate classes with updated mesh
hydro   = GLADS(mesh, results_dir)
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler(mesh, stokes, hydro)

# melt input to hydro model
# print("Interpolating melt rates, taking a while..")
# r = gu.Raster("NETCDF:Greenland_data/MARv3.14-monthly-ERA5_1940_2023.nc:water_input_rate")
# delta = r.res[0]*2
# r.crop([min(meshx)-delta, min(meshy)-delta, max(meshx)+delta, max(meshy)+delta], inplace=True)
# year_0  = 2016 - 1940 # starts in 1940
# n_years = 6
# b_0 = year_0*12
# b_end = b_0 + n_years*12
# i_months = range(b_0,b_end+1)
mm = df.Function(V)
f_melt = VTKFile(results_dir+"m0_per_year.pvd")
# melt = np.zeros((len(H.vector()[:]), len(i_months)))  # will interpolate onto same mesh as H
# for (n,i) in enumerate(i_months):
#     print(n)
#     mm.vector()[:] = r.interp_points((meshx, meshy), band=i) / coupler.rho_w * 12
#     # f_melt.write(mm, time=n)
#     melt[:,n] = mm.vector()[:]
# seasonal = True
# print("... done")

# fenics melt water input
seasonal = True
melt = np.zeros((len(H.vector()[:]), 12))
for i in range(12):
    fenics_smb_file = f"Greenland_data/russel/SMB_fenics/SMB_{i}.h5"
    with CheckpointFile(fenics_smb_file, 'r') as afile:
        mesh_ = afile.load_mesh()
        V_ = df.FunctionSpace(mesh_, "CG", 1)
        # B_ = df.Function(V_).interpolate(afile.load_function(mesh_, "B"))
        # H_ = df.Function(V_).interpolate(afile.load_function(mesh_, "H"))
        SMB_ = df.Function(V).interpolate(afile.load_function(mesh_, "SMB"))
        melt[:,i] = SMB_.vector()[:]

# first time step:
m = df.Function(V)
m.vector()[:] = melt[:,0]

# Convenience functions for calculating the N scale
ones = df.Function(coupler.Q_cg)
ones.vector()[:] = 1.
area = df.assemble(ones*df.dx)
H_mean = df.assemble(H*df.dx)/area
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

f_B = VTKFile(results_dir+"B.pvd").write(B)
f_H = VTKFile(results_dir+"H.pvd").write(H)
f_N = VTKFile(results_dir+"N.pvd")
N   = df.Function(coupler.Q_cg)

# set geometries and variables
coupler.set_geometry(B, H)
hlp.plot_geometry(coupler.B, coupler.H, mesh)
stokes.set_coupler(coupler)
hydro.set_coupler(coupler)
hydro.build_variables()
stokes.build_variables()
hydro.build_forms(m, dt0=get_dt(max(melt[:,0])), e_v=args.e_v, h_r=args.h_r, k_c=args.k_c, k_s=args.k_s, l_c=args.l_c, l_r=args.l_r)
stokes.build_forms(beta2=args.beta2, q=args.q, p=args.p, Nhat=Nhat, Uhat=Uhat)

x, y = df.SpatialCoordinate(mesh)
# hydro.set_initial_phi(0.0)
# hydro.set_initial_S(1*(10-(x+2.3e5)/3e5))
# hydro.set_initial_phi(0.0)
# hydro.set_initial_S(0.1)
chk_file = args.data_directory + "russel/initial_fields_russel_base_melt.h5"
csv_file = args.data_directory + "russel/initial_S_russel_base_melt.csv"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))


solver_params = {#"snes_linesearch_type": "l2",#newton
                 "snes_type":"newtonls",
                 "pc_factor_mat_solver_type": "mumps", # ?
                 "snes_rtol": 1e-3,
                 "snes_atol": 1e0,
                 "snes_max_it": 50,
                 "report": True,
                 "snes_monitor": None,
                 "error_on_nonconvergence": True}

# time stepping and solve
t       = 0.0
d       = 0    # count the days
t_end   = 10
success = True
with df.CheckpointFile(f"{results_dir}/time_series.h5", 'w') as afile:
    afile.save_mesh(mesh)
    i     = 1    # idx for checkpointing

    while (t <= t_end):
        dt = float(hydro.dt.values()[0])
        print(np.max(hydro.m.vector()[:]))
        print("Time = {:.2f} years, dt = {:.1f} hours".format(t, dt*365*24))
        if dt < dt_min:
            # write failure to table
            success = False
            print("Minimal time step reached. Simulation failed.")
            break
        try:
            if seasonal:
                month = (t%1)*12
                month_floor = int(np.floor(month))
                month_ceil = int(np.ceil(month))
                floor_weight = month_ceil - month
                ceil_weight = month - month_floor
                hydro.m.vector()[:] = melt[:,int(month_floor%12)]*floor_weight + melt[:,int(month_ceil%12)]*ceil_weight
                mm.interpolate(hydro.m)
                f_melt.write(mm, time=t)

            df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=solver_params)
            f_N.write(N.interpolate(hydro.N))
            hydro.update_time_variables()
            t += dt
            dt_max = get_dt(np.max(hydro.m.vector()[:]))
            hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
            if int(t*365) >= d+2:
                d = int(t*365)
                hydro.write_variables_pvd(t)
                stokes.write_variables_pvd(t)
                afile.save_function(stokes.Us, idx=i, name="Us")
                afile.save_function(stokes.Ub, idx=i, name="Ub")
                afile.save_function(coupler.U.sub(4), idx=i, name="phi")
                afile.save_function(coupler.U.sub(5), idx=i, name="h")
                i += 1

        except df.exceptions.ConvergenceError:
            # If solver fails, try again with a smaller time step
            hydro.dt.assign(dt*timestep_reduction_fraction)
            print("Convergence not achieved.  Reducing time step to {:.1f} hours and trying again".format(hydro.dt.values()[0]*365*24))

# save end states for future initialization
# chk_file_save = results_dir + "initial_fields_russel_base_melt.h5"
# csv_file_save = results_dir + "initial_S_russel_base_melt.csv"
# hydro.save_end_state(chk_file_save, csv_file_save)

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
# hlp.scatterplt_fields(coupler.U.subfunctions[4:], ["phi", "h", "S"], df.MixedElement(hydro.elements), mesh, results_dir, "russel")
# hlp.scatterplt_fields(coupler.U.subfunctions[0:2], ["ubar_x", "ubar_y"], df.MixedElement(stokes.elements[0:2]), mesh, results_dir, "russel")

# write parameters to table:
hlp.save_params_to_csv(args, params_output_file, success=success)
