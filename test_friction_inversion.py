from firedrake.adjoint import minimize, ReducedFunctional, Control, continue_annotation
import firedrake as df
from models_main.coupled_model import GLADS, SpecFO, Coupler_Flow, Coupler_Flow_Hydro
import models_main.helpers as hlp
import numpy as np
import geoutils as gu

args = hlp.get_args()   # command line arguments

results_dir = "test_inversion/"
data_dir    = args.data_directory

# mesh
mesh_file = data_dir+'russel/russel.msh'
mesh = df.Mesh(mesh_file)
x, y = df.SpatialCoordinate(mesh)

# initiate classes
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler_Flow(mesh, stokes)

# geometry
B, H = hlp.get_topography(mesh, args)

# Convenience functions for calculating the N scale
ones = df.Function(coupler.Q_cg)
ones.dat.data[:] = 1.
area = df.assemble(ones*df.dx)
H_mean = float(df.assemble(H*df.dx)/area)
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

# synthetic friction field
beta2_true = df.Function(coupler.Q_cg)
beta2_true.interpolate(2.8e5+(x+233e3)*2e5/(57e3+233e3))
df.VTKFile(results_dir+"beta2_true.pvd").write(beta2_true)

# set geometries and variables
coupler.set_geometry(B, H)
stokes.set_coupler(coupler)
stokes.build_variables()
stokes.build_forms(beta2=beta2_true, q=1.0, p=1.0, Nhat=Nhat, Uhat=Uhat, N_frac=0.1)

# solve (--> create synthetic data)
solver_params = {"snes_type": "newtonls",#newton
                 "pc_factor_mat_solver_type": "mumps", # ?
                 "snes_rtol": 1e-5,
                 "snes_atol": 1e0,
                 "snes_max_it": 100}
df.solve(coupler.R == 0, coupler.U, solver_parameters=solver_params)
stokes.write_variables_pvd(0)

# add Gaussian noise to synthetic data
noise_level = 0.05  # 5% relative noise
np.random.seed(42)
noise = noise_level * np.random.randn(*stokes.Us.dat.data_ro.shape)
obs_data = stokes.Us.dat.data_ro + noise



#### reset ####
from pyadjoint import get_working_tape
get_working_tape().clear_tape()
continue_annotation()

#### rebuild problem from scratch ####

# mesh
mesh_file = data_dir+'russel/russel.msh'
mesh = df.Mesh(mesh_file)
x, y = df.SpatialCoordinate(mesh)

# initiate classes
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler_Flow(mesh, stokes)

# geometry
B, H = hlp.get_topography(mesh, args)

# Convenience functions for calculating the N scale
ones = df.Function(coupler.Q_cg)
ones.dat.data[:] = 1.
area = df.assemble(ones*df.dx)
H_mean = float(df.assemble(H*df.dx)/area)
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

# set geometries and variables
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler_Flow(mesh, stokes)
coupler.set_geometry(B, H)
stokes.set_coupler(coupler)
stokes.build_variables()
m = df.Function(coupler.Q_cg).interpolate(3e5)
stokes.build_forms(beta2=m, q=1.0, p=1.0, Nhat=Nhat, Uhat=Uhat, N_frac=0.1)

# solve
df.solve(coupler.R == 0, coupler.U, solver_parameters=solver_params)
stokes.write_variables_pvd(1)

# observations on new mesh
Us_obs = df.Function(coupler.Q_cg)
Us_obs.dat.data[:] = obs_data
df.VTKFile(results_dir+"Us_data.pvd").write(Us_obs)

# calculate misfit
Us_model = df.sqrt(coupler.stokes.u(0)**2+coupler.stokes.v(0)**2)
df.VTKFile(results_dir+"Us_model.pvd").write(df.project(Us_model,coupler.Q_cg))
diff = Us_model - Us_obs
J_misfit = df.inner(diff,diff)*df.dx

alpha = 1e-5   # tune this
J_reg = alpha * df.inner(df.grad(m), df.grad(m))*df.dx

# print("Initial J_misfit:", df.assemble(J_reg))
J = df.assemble(J_misfit + J_reg)

Jhat    = ReducedFunctional(J, Control(m))
result = minimize(Jhat) #, options={"maxiter": 100})

df.VTKFile(results_dir+f"beta2_opt_1e{int(np.log10(alpha))}.pvd").write(result)
