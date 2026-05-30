from firedrake.adjoint import minimize, ReducedFunctional, Control, continue_annotation
import firedrake as df
from models_main.coupled_model import GLADS, SpecFO, Coupler_Flow, Coupler_Flow_Hydro
import models_main.helpers as hlp
from plotting.loading_functions import load_vel_obs, load_model_output
import numpy as np
import geoutils as gu

args = hlp.get_args()   # command line arguments

results_dir = "test_inversion/"
data_dir    = args.data_directory

from pyadjoint import get_working_tape
get_working_tape().clear_tape()
continue_annotation()

# mesh
mesh_file = data_dir+'russel/russel.msh'
mesh = df.Mesh(mesh_file)
x, y = df.SpatialCoordinate(mesh)

# get observations
# for now: snapshot:
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
vel_file = "GL_vel_mosaic_Monthly_01Jan19_31Jan19_vv_v05.0.tif"
vel_dates, Us_obs, Us_mask = load_vel_obs(vel_dir, [vel_file], mesh)

Us_obs = Us_obs[0]
Us_mask = Us_mask[0]

df.VTKFile(results_dir+"Us_data.pvd").write(Us_obs)

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

# get N from model output
run_index = 169
idx       = int(365/2)*3
model_file = f"parameter_runs/run_{run_index}/time_series.h5"
_, _, phi_raw, _, _, n_idx = load_model_output(model_file)
phi = df.Function(coupler.Q_cg)
phi.dat.data[:] = phi_raw[:, idx]
N   = df.Function(coupler.Q_cg)
N.interpolate(910*9.81*H-(phi-1000*9.81*B))
df.VTKFile(results_dir+"N.pvd").write(N)

# set geometries and variables
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler_Flow(mesh, stokes)
coupler.set_geometry(B, H)
stokes.set_coupler(coupler)
stokes.build_variables()
# m = df.Function(coupler.Q_cg).interpolate(3e5)
m = df.Function(coupler.Q_cg).interpolate(2.2e5+(x+233e3)*3e5/(57e3+233e3))
stokes.build_forms(beta2=m, q=1.0, p=1.0, Nhat=Nhat, Uhat=Uhat, N=N)

# solve
# solver_params = {"snes_type": "newtonls",
#                  "pc_factor_mat_solver_type": "mumps",
#                  "snes_rtol": 1e-5,
#                  "snes_atol": 1e-3,
#                  "snes_max_it": 100}
solver_params = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_max_it": 50,
    "snes_rtol": 1e-4,
    "snes_atol": 1e0,
    "pc_factor_mat_solver_type": "mumps"
}

try:
    df.solve(coupler.R == 0, coupler.U, solver_parameters=solver_params)
    # calculate misfit
    # stokes.write_variables_pvd(1)
    Us_model = df.sqrt(coupler.stokes.u(0)**2+coupler.stokes.v(0)**2)
    # df.VTKFile(results_dir+"Us_model.pvd").write(df.project(Us_model,coupler.Q_cg))
    diff = Us_mask*(Us_model - Us_obs)
    J_misfit = df.inner(diff,diff)*df.dx
except df.ConvergenceError:
    J = 1e20

alpha = 10**(-1.8)   # tune this
J_reg = alpha * df.inner(df.grad(m), df.grad(m))*df.dx

# print("Initial J_misfit:", df.assemble(J_reg))
J = df.assemble(J_misfit + J_reg)

Jhat    = ReducedFunctional(J, Control(m))
result = minimize(Jhat, method="L-BFGS-B", bounds=(1e5, 8e5)) #, options={"maxiter": 100})

df.VTKFile(results_dir+f"beta2_opt_1e{int(np.log10(alpha))}.pvd").write(result)
with df.CheckpointFile(f"{results_dir}/beta2_opt_1e{int(np.log10(alpha))}.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(result, name="beta2")
