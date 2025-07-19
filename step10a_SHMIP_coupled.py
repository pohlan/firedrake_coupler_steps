import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
from models_main.coupled_model import GLADS, SpecFO, Coupler
import models_main.helpers as hlp
import numpy as np
import pandas as pd

s_per_day = 3600 * 24
results_dir = 'step_10a/'

m          = 5.79e-7
e_v        = 1e-3

# mesh
# mesh = df.Mesh("valley.msh")
# x, y = df.SpatialCoordinate(mesh)

nx, ny = 75, 25
Lx, Ly = 100e3, 20e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
x, y = df.SpatialCoordinate(mesh)

# time stepping
dt0 = 0.01/365
dt_max = 45/365    # s_per_day*45
dt_min = 1e-3/365 #s_per_day*1e-3
timestep_increase_fraction = 1.05
timestep_reduction_fraction = 0.5

# dt0 = s_per_day*0.01
# dt_max = s_per_day*45
# dt_min = s_per_day*1e-3
# timestep_increase_fraction = 1.1
# timestep_reduction_fraction = 0.9

# initiate classes
hydro   = GLADS(mesh, results_dir)
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler(mesh, stokes, hydro)

# geometry
# shmip_suit = "E1"
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

shmip_suit = "A6"
def surface(x,y):
    return 6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1
def bed(x,y):
    return 0

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
stokes.build_forms(beta2=100)
hlp.plot_geometry(coupler.B, coupler.H, mesh)

# for initial state, take steady state solution from a different run
chk_file    = "step_4/initial_fields_A6.h5"
csv_file    = "step_4/initial_S_A6.csv"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))
# hydro.set_initial_S(10 * np.random.rand(len(hydro.U.sub(2).vector()[:])))
# hydro.set_initial_S(20*(1-x/100e3))


solver_params = {"snes_type": "vinewtonrsls",#newton
                 "pc_factor_mat_solver_type": "mumps", # ?
                 "snes_rtol": 1e-2,
                 "snes_atol": 1e-2,
                 "snes_max_it": 300,
                 "report": True,
                 "snes_monitor": None,
                 "error_on_nonconvergence": True}
                 # relaxation_parameter = 0.7   ???

# df.solve(coupler.R == 0, coupler.U, solver_parameters=solver_params)
# stokes.write_variables_pvd(0)

bla = df.Function(coupler.Q_cg)
# bla.interpolate(hydro.N)
N_fl = VTKFile("step_10a/N.pvd")
f_fl = VTKFile("step_10a/f.pvd")

# time stepping and solve
t     = 0.0
t_end = 30
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    print(f"Time = {t} years, dt = {dt*365} days")
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=solver_params)
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
        hydro.write_variables_pvd(t)
        stokes.write_variables_pvd(t)

        N_fl.write(bla.interpolate(hydro.N))
        # f_fl.write(bla.interpolate(hydro.f))
    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
hlp.scatterplt_fields(coupler.U.subfunctions[4:], ["phi", "h", "S"], df.MixedElement(hydro.elements), mesh, results_dir, shmip_suit)
hlp.scatterplt_fields(coupler.U.subfunctions[0:2], ["ubar_x, ubar_y"], df.MixedElement(stokes.elements[0:2]), mesh, results_dir, shmip_suit)
