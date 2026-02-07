import firedrake as df
from models_main.coupled_model import GLADS, Coupler
import models_main.helpers as hlp
from firedrake.checkpointing import CheckpointFile
import numpy as np
import pandas as pd

s_per_day = 3600 * 24
s_per_year = s_per_day*365
results_dir = 'step_4/'

# mesh
nx, ny = 75, 25
Lx, Ly = 100e3, 20e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
x, y = df.SpatialCoordinate(mesh)

# shmip
shmip_suit = "A6"
shmip_m = {"A1" : 7.93e-11,
           "A2" : 1.59e-9,
           "A3" : 5.79e-9,
           "A4" : 2.5e-8,
           "A5" : 4.5e-8,
           "A6" : 5.79e-7}
m = shmip_m[shmip_suit]*s_per_year

# geometry
def surface(x,y):
    return 6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1
def bed(x,y):
    return 0

# time stepping
dt0 = 0.01/365
dt_max = 5/365
dt_min = 1e-3/365
timestep_increase_fraction = 1.05
timestep_reduction_fraction = 0.9

# hydro object
hydro = GLADS(mesh, results_dir)
coupler = Coupler(mesh,hydro)

# set geometries and variables
B = df.Function(coupler.Q_cg).interpolate(bed(x,y))
H = df.Function(coupler.Q_cg).interpolate(surface(x,y)-bed(x,y))
coupler.set_geometry(B, H)
hydro.set_coupler(coupler)
hydro.build_variables()
hydro.build_forms(m, dt0=dt0, e_v=0, u_b=1e-6*s_per_year, h_r=0.1, l_r=2, l_c=2, k_s=0.005, k_c=0.1)
hlp.plot_geometry(coupler.B, coupler.H, mesh)

# for initial state, take steady state solution from a different run
chk_file    = results_dir + "initial_fields_A5.h5"
csv_file    = results_dir + "initial_S_A5.csv"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))
# hydro.set_initial_S(10 * np.random.rand(len(hydro.U.sub(2).vector()[:])))
# hydro.set_initial_S(60*(1-x/100e3))
# hydro.set_initial_S(1.0)

# solver options
par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e0,
       "snes_max_it": 100,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

# time stepping and solve
t     = 0.0
t_end = 50
d     = 0   # days
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    print("Time = {:.2f} years, dt = {:.1f} days".format(t, dt*365))
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=par)
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
        if int(t*365) >= d+2:
            d = int(t*365)
            hydro.write_variables_pvd(t)

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# save end result such that it can serve as initial field for another simulation
# if shmip_suit == "A1":
chk_file_save = results_dir + "initial_fields_"+shmip_suit+".h5"
csv_file_save = results_dir + "initial_S_"+shmip_suit+".csv"
hydro.save_end_state(chk_file_save, csv_file_save)

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
hlp.scatterplt_fields(coupler.U.subfunctions, ["phi", "h", "S"], coupler.E_V, mesh, results_dir, shmip_suit)

# test against GlaDS-matlab SHMIP results
# fl = "SHMIP_results/mw/"+shmip_suit+"_mw.nc"
# hlp.diff_to_glads_matlab(hydro, fl)
