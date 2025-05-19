import firedrake as df
from GlaDS_main.hydro_class import GLADS
import GlaDS_main.helpers as hlp
import numpy as np

s_per_day = 3600 * 24
results_dir = 'step_5/'

# shmip
shmip_suit = "E4"
m = 1.158e-6 # E suit

# mesh
mesh = df.Mesh("valley.msh")

# time stepping
dt0 = s_per_day*0.001
dt_max = s_per_day*15
dt_min = s_per_day*1e-5
timestep_increase_fraction = 1.01
timestep_reduction_fraction = 0.5

# hydro object
hydro = GLADS(mesh, results_dir)
hydro.build_variables(m, dt0)

# # A suites
# def surface(x,y):
#     6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1
# def bed(x,y):
#     0
#
# hydro.set_geometry(surface, bed)
# shmip_m = {"A1" : 7.93e-11,
#            "A2" : 1.59e-9,
#            "A3" : 5.79e-9,
#            "A4" : 2.5e-8,
#            "A5" : 4.5e-8,
#            "A6" : 5.79e-7}
# m = shmip_m[shmip_suit]

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

hydro.set_geometry(surface, bed)
hlp.plot_geometry(hydro)

hydro.set_initial_S(10 * np.random.rand(len(hydro.U.sub(2).vector()[:])))

# solver options
par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e-3,
       "snes_max_it": 100,
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

        # Doug's trick to save Q
        # df.solve(F_Q == 0, dQ)
        # outfile_Q.write(df.project(dQ, V_S, name="Q"))

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# save end result such that it can serve as initial field for another simulation
# if shmip_suit == "A1":
# chk_file_save = results_dir + "initial_fields_"+shmip_suit+".h5"
# hydro.save_end_state(chk_file_save)

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
hlp.scatterplt_fields(hydro, shmip_suit)

# test against GlaDS-matlab SHMIP results
fl = "SHMIP_results/mw/"+shmip_suit+"_mw.nc"
hlp.diff_to_glads_matlab(hydro, fl)
