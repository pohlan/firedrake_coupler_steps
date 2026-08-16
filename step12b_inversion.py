import os
os.environ['OMP_NUM_THREADS'] = '1'
from firedrake.adjoint import minimize, ReducedFunctional, Control, continue_annotation
import firedrake as df
from models_main.coupled_model import GLADS, SpecFO, Coupler_Flow, Coupler_Flow_Hydro
import models_main.helpers as hlp
from plotting.loading_functions import *
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
# vel_file = "GL_vel_mosaic_Monthly_01Jan21_31Jan21_vv_v05.0.tif"
vel_file = "GL_vel_mosaic_Monthly_01Nov18_30Nov18_vv_v05.0.tif"
vel_dates, Us_obs, Us_mask = load_obs_FunctionSpace(vel_dir, [vel_file], mesh)
print(vel_dates)
Us_obs = Us_obs[0]
Us_mask = Us_mask[0]

# initiate classes
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler_Flow(mesh, stokes)

# geometry
B, H = hlp.get_topography(mesh, args)
S    = df.Function(coupler.Q_cg).interpolate(B+H)
df.VTKFile(results_dir+"S.pvd").write(S)

# adjust mask??
# Us_mask.dat.data[np.where((S.dat.data_ro < 500) | (S.dat.data_ro > 1100))] = 0
df.VTKFile(results_dir+"Us_data.pvd").write(Us_obs)
df.VTKFile(results_dir+"Us_mask.pvd").write(Us_mask)

# Convenience functions for calculating the N scale
ones = df.Function(coupler.Q_cg)
ones.dat.data[:] = 1.
area = df.assemble(ones*df.dx)
H_mean = float(df.assemble(H*df.dx)/area)
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

# get N from model output
run_index = args.run_index
idx       = int(365/2*4.9)   # 4.87 = November 15, 2018; 8.041: January 2022 (new1)
model_file = f"parameter_runs/run_{run_index}/time_series.h5"
_, _, phi_raw, _, _, _, n_idx, N_raw = load_model_output(model_file)
N = df.Function(coupler.Q_cg)
N.dat.data[:] = N_raw[:, idx]
# phi = df.Function(coupler.Q_cg)
# phi.dat.data[:] = phi_raw[:, idx]
# N   = df.Function(coupler.Q_cg)
# N.interpolate(910*9.81*H-(phi-1000*9.81*B))
N.dat.data[:] = np.maximum(N.dat.data[:], 1e5)
N.dat.data[:] = np.minimum(N.dat.data[:], 910*9.81*H.dat.data[:])

df.VTKFile(results_dir+"N.pvd").write(N)

# set geometries and variables
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler_Flow(mesh, stokes)
coupler.set_geometry(B, H)
stokes.set_coupler(coupler)
stokes.build_variables()
params = hlp.get_params_from_input_file(run_index)
pval   = params['p']
qval   = params['q']
beta2val   = params['beta2']
m = df.Function(coupler.Q_cg).interpolate(beta2val)
# m = df.Function(coupler.Q_cg).interpolate(2.2e5+(x+233e3)*3e5/(57e3+233e3))
stokes.build_forms(beta2=m, q=qval, p=pval, Nhat=Nhat, Uhat=Uhat, N=N)

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
    Us_model = df.sqrt(coupler.stokes.u(0)**2+coupler.stokes.v(0)**2)
    # df.VTKFile(results_dir+"Us_model.pvd").write(df.project(Us_model,coupler.Q_cg))
    diff = Us_mask*(Us_model - Us_obs)
    J_misfit = df.inner(diff,diff)*df.dx
except df.ConvergenceError:
    J = 1e20

alpha = 10**(-2)   # tune this
J_reg = alpha * df.inner(df.grad(m), df.grad(m))*df.dx

# print("Initial J_misfit:", df.assemble(J_reg))
J = df.assemble(J_misfit + J_reg)

Jhat    = ReducedFunctional(J, Control(m))
result = minimize(Jhat, method="L-BFGS-B", bounds=(1e5, 6e6)) #, options={"maxiter": 100})

df.VTKFile(results_dir+f"beta2_opt_1e{round(np.log10(alpha),ndigits=1)}_run{run_index}.pvd").write(result)
with df.CheckpointFile(f"{results_dir}/beta2_opt_1e{round(np.log10(alpha),ndigits=1)}_run{run_index}.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(result, name="beta2")





stokes.build_forms(beta2=result, q=qval, p=pval, Nhat=Nhat, Uhat=Uhat, N=N)
df.solve(coupler.R == 0, coupler.U, solver_parameters=solver_params)
stokes.write_variables_pvd(2)

Us_model = df.sqrt(coupler.stokes.u(0)**2+coupler.stokes.v(0)**2)
diff = df.Function(coupler.Q_cg).interpolate(Us_mask*(Us_model - Us_obs))
df.VTKFile(results_dir+f"diff_p{pval}_q{qval}.pvd").write(diff)

print(f"p={pval}, q={qval}")
print(np.median(abs(diff.dat.data_ro)))
print(np.std(diff.dat.data_ro))

