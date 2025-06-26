import firedrake as df
from GlaDS_main.hydro_class import GLADS
import GlaDS_main.helpers as hlp
from firedrake.checkpointing import CheckpointFile
import numpy as np
import pandas as pd

s_per_hour  = 3600
s_per_day   = s_per_hour * 24
s_per_year  = s_per_day * 365
results_dir = 'step_7/'

# mesh
mesh = df.Mesh("valley.msh")
x, y = df.SpatialCoordinate(mesh)

# shmip
shmip_suit = "F5"
shmip_DT = {"F1" : -6,
            "F2" : -3,
            "F3" :  0,
            "F4" :  3,
            "F5" :  6}
DT = shmip_DT[shmip_suit]

# E suites geometry
para_bench = 0.05
shmip_para = {"E1":  0.05,
              "E2":  0.0 ,
              "E3": -0.1 ,
              "E4": -0.5 ,
              "E5": -0.7 }
para = shmip_para["E1"]
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

# seasonal melt water input
def temp(t):
    return -16*df.cos(2*df.pi/s_per_year*t)- 5 + DT
def runoff(t):
    lr    = -0.0075
    DDF   = 0.01/86400
    basal = 7.93e-11
    zs    = surface(x,y)
    return df.max_value(0, (zs*lr+temp(t))*DDF) + basal
m = runoff(0.0)
e_v        = 1e-3

# time stepping
dt0    = s_per_hour*5
# dt_max = s_per_hour*5
dt_min = s_per_hour*1e-2
timestep_increase_fraction = 1.1
timestep_reduction_fraction = 0.5
def get_dt(m):
    return max(s_per_hour*5, s_per_hour*15 + s_per_hour*(0.5-15)/(8.1e-7-7.93e-11) * (m-7.93e-11))

# hydro object
hydro = GLADS(mesh, results_dir)
hydro.build_variables(m, dt0, surface, bed, e_v)

hlp.plot_geometry(hydro)

# for initial state, take steady state solution from a different run  (make sure it is the steady state for E1 and m=7.93e-11 not the default m)
chk_file    =  "step_5/initial_fields_E1.h5"
csv_file    =  "step_5/initial_S_E1.csv"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))

# solver options
par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e-4,
       "snes_max_it": 500,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

# time stepping and solve
t     = 0.0
d     = 0    # count the days
t_end = s_per_year*6
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    print([t / s_per_year, dt / s_per_hour])
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(hydro.F == 0, hydro.U, bcs=hydro.bcs, solver_parameters=par)
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(get_dt(np.max(hydro.m.vector()[:])))
        hydro.m.interpolate(runoff(t))
        if int(t / s_per_day) >= d+10:
            d = int(t / s_per_day)
            print(np.max(hydro.m.vector()[:]))
            print(d)
            hydro.write_variables_pvd(d)

        # Doug's trick to save Q
        # df.solve(F_Q == 0, dQ)
        # outfile_Q.write(df.project(dQ, V_S, name="Q"))

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))
