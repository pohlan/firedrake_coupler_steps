import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
# from models_main.hydro_class import GLADS, Coupler
from models_main.coupled_model import GLADS, SpecFO, Coupler
import models_main.helpers as hlp
import numpy as np
import pandas as pd

import geoutils as gu
from firedrake.__future__ import interpolate

s_per_day = 3600 * 24
results_dir = 'step_10b/results/'
data_dir    = 'Greenland_data/'

m          = 3e-13

# mesh
# mesh_file = data_dir+'western_med_v2.msh'
mesh_file = data_dir+'russel/russel.msh'
mesh = df.Mesh(mesh_file)

# time stepping
dt0 = 0.05/365
dt_max = 45/365
dt_min = 1e-3/365
timestep_increase_fraction = 1.1
timestep_reduction_fraction = 0.5

# geometry
V = df.FunctionSpace(mesh, "CG", 1)
v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
X = df.assemble(interpolate(mesh.coordinates,v_dg))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]
B = df.Function(V)
H = df.Function(V)

sig = 1
# r_bed = gu.Raster(f"{data_dir}BedMachineGreenland-v5_bed_smooth_sig{sig}.nc")
r_bed = gu.Raster(f"{data_dir}BedMachineGreenland-v5_bed_smooth.nc")
B.dat.data[:] = r_bed.interp_points((meshx, meshy))
# r_thk = gu.Raster(f"{data_dir}BedMachineGreenland-v5_thickness_smooth_sig{sig}.nc")
r_thk = gu.Raster(f"{data_dir}BedMachineGreenland-v5_thickness_smooth.nc")
H.dat.data[:] = r_thk.interp_points((meshx, meshy))

# make bed elevation and thickness the same at bc points of individual outlets
bc_nodes = V.boundary_nodes(1)
for i in range(0,len(bc_nodes),2):
    nodes = bc_nodes[i:i+2]
    B.vector()[nodes] = np.mean(B.vector()[nodes])
    H.vector()[nodes] = np.mean(H.vector()[nodes])

# set minimum ice thickness to 10
thklim = 0
thklim = 10
Htemp = H.vector().get_local()
Htemp[Htemp<thklim] = thklim
H.vector().set_local(Htemp)

# initiate classes with updated mesh
hydro   = GLADS(mesh, results_dir)
stokes  = SpecFO(mesh, results_dir)
# coupler = Coupler(mesh, hydro)
coupler = Coupler(mesh, stokes, hydro)

# Convenience functions for calculating the N scale
ones = df.Function(coupler.Q_cg)
ones.vector()[:] = 1.
area = df.assemble(ones*df.dx)
H_mean = df.assemble(H*df.dx)/area
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

# melt, take first time step from MAR
# m = df.Function(V)
# r = gu.Raster("NETCDF:Greenland_data/MARv3.14-monthly-ERA5_1940_2023.nc:water_input_rate")
# delta = r.res[0]*2
# r.crop([min(meshx)-delta, min(meshy)-delta, max(meshx)+delta, max(meshy)+delta], inplace=True)
# b_0 = (2016 - 1940)*12+2
# m.vector()[:] = r.interp_points((meshx, meshy), band=b_0) / coupler.rho_w * 12

# fenics melt water input
# m = df.Function(V)
# fenics_smb_file = f"Greenland_data/russel/SMB_fenics/SMB_0.h5"
# with CheckpointFile(fenics_smb_file, 'r') as afile:
#     mesh_ = afile.load_mesh()
#     V_ = df.FunctionSpace(mesh_, "CG", 1)
#     SMB_ = df.Function(V).interpolate(afile.load_function(mesh_, "SMB"))
#     m.vector()[:] = SMB_.vector()[:]

# f_melt = VTKFile(results_dir+"m0_per_year.pvd").write(m)
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
# hydro.build_forms(m, dt0=dt0, e_v=1e-4, h_r=1.0, k_c=0.5, k_s=0.03, l_c=10.0, l_r=5.0, transition=True)
hydro.build_forms(m, dt0=dt0, e_v=1e-3, h_r=0.1, k_c=0.05, k_s=5e-4, l_c=10.0, l_r=2.0, transition=False)
stokes.build_forms(beta2=1e6, q=0.5, p=0.5, Nhat=Nhat, Uhat=Uhat)

# hydro.set_initial_phi(0.0)
# hydro.set_initial_S(0.01)
# chk_file = f"step_10b/results/initial_fields_russel_base_melt_smooth_new.h5"
# csv_file = f"step_10b/results/initial_S_russel_base_melt_smooth_new.csv"
# chk_file = args.data_directory + "russel/initial_fields_russel_base_melt.h5"
# csv_file = args.data_directory + "russel/initial_S_russel_base_melt.csv"
# with CheckpointFile(chk_file, 'r') as afile:
#     mesh_ = afile.load_mesh()
#     hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
#     hydro.set_initial_h(afile.load_function(mesh_, "h"))
# hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))


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
            stokes.write_variables_pvd(t)

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        coupler.U.sub(4).assign(hydro.phi0)
        coupler.U.sub(5).assign(hydro.h0)
        coupler.U.sub(6).assign(hydro.S0)
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# save end states for future initialization
# chk_file_save = results_dir + f"initial_fields_russel_base_melt_smooth_sig{sig}.h5"
chk_file_save = results_dir + f"initial_fields_russel_base_melt_smooth_new_coupled.h5"
# csv_file_save = results_dir + f"initial_S_russel_base_melt_smooth_sig{sig}.csv"
csv_file_save = results_dir + f"initial_S_russel_base_melt_smooth_new_coupled.csv"
hydro.save_end_state(chk_file_save, csv_file_save)

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
# hlp.scatterplt_fields(coupler.U.subfunctions[4:], ["phi", "h", "S"], df.MixedElement(hydro.elements), mesh, results_dir, "russel")
# hlp.scatterplt_fields(coupler.U.subfunctions[0:2], ["ubar_x", "ubar_y"], df.MixedElement(stokes.elements[0:2]), mesh, results_dir, "russel")
