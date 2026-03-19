import firedrake as df
from models_main.coupled_model import SHAKTI, Coupler_Hydro
import models_main.helpers as hlp
from firedrake.checkpointing import CheckpointFile
import numpy as np

s_per_day = 3600 * 24
s_per_year = s_per_day*365
results_dir = 'step_11a/'

# mesh
nx, ny = 100, 25
Lx, Ly = 100e3, 20e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0, diagonal="crossed")
# mesh = df.Mesh("valley.msh")
x, y = df.SpatialCoordinate(mesh)

# shmip
shmip_suit = "A1"
shmip_m = {"A1" : 7.93e-11,
           "A2" : 1.59e-9,
           "A3" : 5.79e-9,
           "A4" : 2.5e-8,
           "A5" : 4.5e-8,
           "A6" : 5.79e-7}
m = shmip_m[shmip_suit]*s_per_year
# m = 0.05

# geometry
def surface(x,y):
    return 6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1
def bed(x,y):
    return 0


# E suites geometry
# m = 1.158e-6
# shmip_suit = "E5"
# para_bench = 0.05
# shmip_para = {"E1":  0.05,
#               "E2":  0.0 ,
#               "E3": -0.1 ,
#               "E4": -0.5 ,
#               "E5": -0.7 }
# para = shmip_para[shmip_suit]
# def surface(x,y):
#     return 100*(x+200)**(1/4) + 1/60*x - 2e10**(1/4) + 1
# def f(x,para):
#     return (surface(6e3,0) - para*6e3)/6e3**2 * x**2 + para*x
# def g(y):
#     return 0.5e-6 * abs(y)**3
# def h(x,para):
#     return (-4.5*x/6e3 + 5) * (surface(x,0)-f(x, para)) / (surface(x,0)-f(x, para_bench)+1e-15)
# def bed(x,y):
#     return f(x,para) + g(y) * h(x,para)

# time stepping
# dt0 = s_per_day*0.005
# dt_max = s_per_day*45
# dt_min = s_per_day*1e-5
# timestep_increase_fraction = 1.1
# timestep_reduction_fraction = 0.5

dt0    = 0.01/365
dt_max = 45/365
dt_min = 1e-3/365
timestep_increase_fraction = 1.1
timestep_reduction_fraction = 0.5
day = 1/365
hour = day/24

# hydro object
hydro   = SHAKTI(mesh, results_dir)
coupler = Coupler_Hydro(mesh, hydro)

B = df.Function(coupler.Q_cg).interpolate(bed(x,y))
H = df.Function(coupler.Q_cg).interpolate(surface(x,y)-bed(x,y))

coupler.set_geometry(B, H)
hydro.set_coupler(coupler)
hydro.build_variables()
hydro.build_forms(m=m, dt0=dt0, e_v=0, h_r=0.1, l_r=2, u_b=1e-6*s_per_year, omega=1e-3)
hydro.write_variables_pvd(0)

coupler.nu.assign(1.787e-6)

# solver options
par = {"snes_type": "newtonls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-3,
       "snes_atol": 1e0,
       "snes_max_it": 200,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

# time stepping and solve
t     = 0.0
# t_end = s_per_day*365*30
t_end = 30
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    # print("Time = {:.2f} years, dt = {:.1f} days".format(t / (3600*24*365), dt / s_per_day))
    print("Time = {:.2f} years, dt = {:.1f} days".format(t, dt*365))
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=par)
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
