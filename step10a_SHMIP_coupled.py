import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
from models_main.hydro_class_coupled import GLADS
from models_main.SpecFO import SpecFO, Coupler
import models_main.helpers as hlp
# import numpy as np
# import pandas as pd

# import rasterio as rio
# from firedrake.__future__ import interpolate

s_per_day = 3600 * 24
results_dir = 'step_10a/'

m          = 1e-11
e_v        = 1e-2

# mesh
mesh = df.Mesh("valley.msh")
x, y = df.SpatialCoordinate(mesh)

# time stepping
dt0 = s_per_day*0.01
dt_max = s_per_day*45
dt_min = s_per_day*1e-3
timestep_increase_fraction = 1.01
timestep_reduction_fraction = 0.9

# initiate classes
hydro   = GLADS(mesh, results_dir)
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler(mesh, stokes, hydro)

# geometry
shmip_suit = "E1"
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

B = df.Function(coupler.Q_cg).interpolate(bed(x,y))
H = df.Function(coupler.Q_cg).interpolate(surface(x,y)-bed(x,y))

# set minimum ice thickness to 10
thklim = 0
thklim = 10
Htemp = H.vector().get_local()
Htemp[Htemp<thklim] = thklim
# Htemp[np.isnan(Htemp)] = thklim
H.vector().set_local(Htemp)

# set geometries and variables
coupler.set_geometry(B, H)
stokes.set_coupler(coupler)
hydro.set_coupler(coupler)
hydro.build_variables()
stokes.build_variables()
hydro.build_forms(m, dt0=dt0, e_v=e_v)
stokes.build_forms()
hlp.plot_geometry(coupler.B, coupler.H, mesh)

# par = {"snes_type": "vinewtonrsls",
#        "pc_factor_mat_solver_type": "mumps",
#        "snes_rtol": 1e-5,
#        "snes_atol": 1e-4,
#        "snes_max_it": 300,
#        "report": True,
#        "snes_monitor": None,
#        "error_on_nonconvergence": True}


solver_params = {"snes_type": "vinewtonrsls",#newton
                 "pc_factor_mat_solver_type": "mumps", # ?
                 "snes_rtol": 1e-3,
                 "snes_atol": 1e0,
                 "snes_max_it": 100,
                 "report": True,
                 "snes_monitor": None,
                 "error_on_nonconvergence": True}
                 # relaxation_parameter = 0.7   ???
# problem = df.NonlinearVariationalProblem(coupler.R,coupler.U,bcs=hydro.bcs)
# solver  = df.NonlinearVariationalSolver(problem, solver_parameters=solver_params)


# time stepping and solve
t     = 0.0
t_end = s_per_day*365*10
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    # dt = s_per_day*0.1
    print([t / (3600*24*365), dt / s_per_day])
    # print(min(dt*timestep_increase_fraction,dt_max))
    # if dt < dt_min:
    #     print("Minimal time step reached. Simulation failed.")
    #     break
    # try:
    # problem = df.NonlinearVariationalProblem(coupler.R,coupler.U,bcs=hydro.bcs)
    # solver  = df.NonlinearVariationalSolver(problem, solver_parameters=solver_params)
    # solver.solve()
    df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=solver_params)
    hydro.update_time_variables()
    t += dt
    hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
    hydro.write_variables_pvd(t)
    stokes.write_variables_pvd(t)

    # except df.exceptions.ConvergenceError:
    #     # If solver fails, try again with a smaller time step
    #     hydro.dt.assign(dt*timestep_reduction_fraction)
    #     print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
hlp.scatterplt_fields(hydro, "E1")
