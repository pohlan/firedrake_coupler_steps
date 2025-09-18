import firedrake as df
from models_main.hydro_class import SHAKTI
import models_main.helpers as hlp
from firedrake.checkpointing import CheckpointFile
import numpy as np

s_per_day = 3600 * 24
results_dir = 'step_11/'

# mesh
nx, ny = 75, 25
Lx, Ly = 100e3, 20e3
# mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
mesh = df.Mesh("valley.msh")
x, y = df.SpatialCoordinate(mesh)

# shmip
# shmip_suit = "A6"
# shmip_m = {"A1" : 7.93e-11,
#            "A2" : 1.59e-9,
#            "A3" : 5.79e-9,
#            "A4" : 2.5e-8,
#            "A5" : 4.5e-8,
#            "A6" : 5.79e-7}
# m = shmip_m[shmip_suit]

# # geometry
# def surface(x,y):
#     return 6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1
# def bed(x,y):
#     return 0


# E suites geometry
m = 1.158e-6
shmip_suit = "E5"
para_bench = 0.05
shmip_para = {"E1":  0.05,
              "E2":  0.0 ,
              "E3": -0.1 ,
              "E4": -0.5 ,
              "E5": -0.7 }
para = shmip_para[shmip_suit]
def surface(x,y):
    return 100*(x+200)**(1/4) + 1/60*x - 2e10**(1/4) + 1
def f(x,para):
    return (surface(6e3,0) - para*6e3)/6e3**2 * x**2 + para*x
def g(y):
    return 0.5e-6 * abs(y)**3
def h(x,para):
    return (-4.5*x/6e3 + 5) * (surface(x,0)-f(x, para)) / (surface(x,0)-f(x, para_bench)+1e-15)
def bed(x,y):
    return f(x,para) + g(y) * h(x,para)

# time stepping
dt0 = s_per_day*0.005
dt_max = s_per_day*45
dt_min = s_per_day*1e-5
timestep_increase_fraction = 1.1
timestep_reduction_fraction = 0.5

# hydro object
hydro = SHAKTI(mesh, results_dir)
hydro.build_variables(m, dt0, surface(x,y)-bed(x,y), bed(x,y), e_v_=1e-3)

hlp.plot_geometry(hydro.B, hydro.H, mesh)


# for initial state, take steady state solution from a different run
chk_file    = results_dir + "initial_fields_E2.h5"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_h_w(afile.load_function(mesh_, "h_w"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
    hydro.set_initial_K(afile.load_function(mesh_, "K"))

# solver options
par = {"snes_type": "newtonls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e-5,
       "snes_max_it": 1000,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

# time stepping and solve
t     = 0.0
t_end = s_per_day*365*100
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    print([t / (3600*24*365), dt / s_per_day])
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(hydro.F == 0, hydro.U, bcs=hydro.bcs, solver_parameters=par)
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
        hydro.write_variables_pvd(t)

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# save end result such that it can serve as initial field for another simulation
# if shmip_suit == "A1":
chk_file_save = results_dir + "initial_fields_"+shmip_suit+".h5"
hydro.save_end_state(chk_file_save)
