import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
from firedrake.checkpointing import CheckpointFile
from models_main.coupled_model import GLADS, SHAKTI, SpecFO, Coupler_Flow_Hydro, Coupler_Hydro
import models_main.helpers as hlp
import numpy as np
import geoutils as gu
import pandas as pd

s_per_day = 3600 * 24
results_dir = 'step_10b/results/'
data_dir    = 'Greenland_data/'

# parameters
m          = 0.003 # constant melt input (m/yr)
sig        = 10     # amount of smoothing for bed/thickness input (0=no smoothing)

# mesh
mesh_file = data_dir+'russel/russel_hole.msh'
mesh = df.Mesh(mesh_file)

# time stepping
dt0 = 0.05/365
dt_max = 45/365
dt_min = 1e-3/365
timestep_increase_fraction = 1.05
timestep_reduction_fraction = 0.5

# geometry
V = df.FunctionSpace(mesh, "CG", 1)
v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
X = df.assemble(df.interpolate(mesh.coordinates,v_dg))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]
B = df.Function(V)
H = df.Function(V)


if sig==0:
    r_bed = gu.Raster(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:bed")
    r_thk = gu.Raster(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:thickness")
else:  # higher sigma == more smoothing
    r_bed = gu.Raster(f"{data_dir}BedMachineGreenland-v5_bed_smooth_sig{sig}.nc")
    r_thk = gu.Raster(f"{data_dir}BedMachineGreenland-v5_thickness_smooth_sig{sig}.nc")
# interpolate onto mesh
B.dat.data[:] = r_bed.interp_points((meshx, meshy), as_array=True)
H.dat.data[:] = r_thk.interp_points((meshx, meshy), as_array=True)

# make bed elevation and thickness the same at bc points of individual outlets
bc_nodes = V.boundary_nodes(1)
for i in range(0,len(bc_nodes),2):
    nodes = bc_nodes[i:i+2]
    B.dat.data[nodes] = np.mean(B.dat.data_ro[nodes])
    H.dat.data[nodes] = np.mean(H.dat.data_ro[nodes])

# set minimum ice thickness to 10
thklim = 0
thklim = 10
Htemp = H.dat.data[:]
Htemp[Htemp<thklim] = thklim
H.dat.data[:] = Htemp

# initiate classes with updated mesh
hydro   = GLADS(mesh, results_dir)
coupler = Coupler_Hydro(mesh, hydro)

# Convenience functions for calculating the N scale
ones = df.Function(coupler.Q_cg)
ones.dat.data[:] = 1.
area = df.assemble(ones*df.dx)
H_mean = df.assemble(H*df.dx)/area
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

f_B = df.VTKFile(results_dir+"B.pvd").write(B)
f_H = df.VTKFile(results_dir+"H.pvd").write(H)
f_N = df.VTKFile(results_dir+"N.pvd")
N   = df.Function(coupler.Q_cg)

# set geometries and variables
coupler.set_geometry(B, H)
hlp.plot_geometry(coupler.B, coupler.H, mesh)
hydro.set_coupler(coupler)
hydro.build_variables()
hydro.build_forms(m=m, dt0=dt0, e_v=1e-4, h_r=0.5, k_c=0.2, k_s=0.002, l_c=10, l_r=10, transition=True, u_b=100, omega=0.1)

#########################################
# solve for the hydro-only steady state #
#########################################

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
t     = 0.0
d     = 0    # count the days
t_end = 20
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    print(f"Time = {t} years, dt = {dt*365} days")
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=solver_params)
        f_N.write(N.interpolate(hydro.N))
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
        if int(t*365) >= d+10:
            d = int(t*365)
            hydro.write_variables_pvd(t)

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# save end states for initialization of coupled run
chk_file_hydro = results_dir + f"initial_fields_russel_hydro_sig{sig}.h5"
csv_file_hydro = results_dir + f"initial_S_russel_hydro_sig{sig}.csv"
hydro.save_end_state(chk_file_hydro, csv_file_hydro)

######################################
# solve for the coupled steady-state #
######################################

# initiate classes
hydro   = GLADS(mesh, results_dir)
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler_Flow_Hydro(mesh, stokes, hydro)

# set geometries and variables
coupler.set_geometry(B, H)
stokes.set_coupler(coupler)
hydro.set_coupler(coupler)
hydro.build_variables()
stokes.build_variables()
stokes.build_forms(beta2=1e6, q=1.0, p=1.2, Nhat=Nhat, Uhat=Uhat)
hydro.build_forms(m=m, dt0=dt0, e_v=1e-4, h_r=0.5, k_c=0.2, k_s=0.002, l_c=10, l_r=10, transition=True, omega=0.1)

# take initial state from hydro initialization
with CheckpointFile(chk_file_hydro, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file_hydro).S))

# time stepping and solve
t     = 0.0
d     = 0    # count the days
t_end = 20
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    print(f"Time = {t} years, dt = {dt*365} days")
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=solver_params)
        f_N.write(N.interpolate(hydro.N))
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
        if int(t*365) >= d+10:
            d = int(t*365)
            hydro.write_variables_pvd(t)
            stokes.write_variables_pvd(t)

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# save end states for initialization of seasonal simulations
chk_file_save = results_dir + f"initial_fields_russel_coupled_sig{sig}.h5"
csv_file_save = results_dir + f"initial_S_russel_coupled_sig{sig}.csv"
hydro.save_end_state(chk_file_save, csv_file_save)
