import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
from models_main.hydro_class import GLADS
import models_main.helpers as hlp
import numpy as np
import pandas as pd

s_per_day = 3600 * 24
results_dir = 'step_5/'

# shmip
shmip_suit = "E5"
m          = 1.158e-6
e_v        = 1e-3

# mesh
mesh = df.Mesh("valley.msh")
x, y = df.SpatialCoordinate(mesh)

# time stepping
dt0 = s_per_day*0.01
dt_max = s_per_day*45
dt_min = s_per_day*1e-3
timestep_increase_fraction = 1.01
timestep_reduction_fraction = 0.9


# E suites geometry
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

# hydro object
hydro = GLADS(mesh, results_dir)
hydro.build_variables(m, dt0, surface(x,y)-bed(x,y), bed(x,y), e_v)
hlp.plot_geometry(hydro)


chk_file    =  "step_5/initial_fields_E3.h5"
csv_file    =  "step_5/initial_S_E3.csv"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))

# hydro.set_initial_phi(0.0) # default doesn't work here
# hydro.set_initial_S(10 * np.random.rand(len(hydro.U.sub(2).vector()[:])))
# def S_init(x):
#     return 20 * (1-x/6e3) * np.random.rand()
# hydro.set_initial_S(S_init(x))
# hydro.set_initial_S(20*(1-x/6e3))  # 20 * .. worked for E1-E3

# solver options
par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e-4,
       "snes_max_it": 1000,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

bla = df.Function(hydro.V_phi)
bla.interpolate(hydro.N)
print(min(bla.vector()[:]))
bla_fl = VTKFile("N.pvd")

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

        # Doug's trick to save Q
        # df.solve(F_Q == 0, dQ)
        # outfile_Q.write(df.project(dQ, V_S, name="Q"))

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

from firedrake.pyplot import tripcolor, triplot
import matplotlib.pyplot as plt
fig, axes = plt.subplots()
cl = tripcolor(bla, axes=axes)
fig.colorbar(cl)
plt.savefig("bla.jpg")

# save end result such that it can serve as initial field for another simulation
# if shmip_suit == "A1":
chk_file_save = results_dir + "initial_fields_"+shmip_suit+".h5"
csv_file_save = results_dir + "initial_S_"+shmip_suit+".csv"
hydro.save_end_state(chk_file_save, csv_file_save)

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
hlp.scatterplt_fields(hydro, shmip_suit)

# test against GlaDS-matlab SHMIP results
fl = "SHMIP_results/mw/"+shmip_suit+"_mw.nc"
hlp.diff_to_glads_matlab(hydro, fl)
