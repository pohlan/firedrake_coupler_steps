import firedrake as df
from models_main.coupled_model import SHAKTI, GLADS, Coupler_Hydro
# from models_main.hydro_class import SHAKTI
import models_main.helpers as hlp
from firedrake.checkpointing import CheckpointFile
import numpy as np

s_per_day = 3600 * 24
results_dir = 'step_11b/'

# mesh
nx, ny = 75, 25
Lx, Ly = 100e3, 20e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
# mesh = df.Mesh("valley.msh")
x, y = df.SpatialCoordinate(mesh)

# time stepping
dt0 = 0.5/365
dt_max = 40/(365*24)
dt_min = 1e-3/365
timestep_increase_fraction = 1.05
timestep_reduction_fraction = 0.5
day = 1/365
hour = day/24
def get_dt(m):
    return max(10.0*hour, 30*hour + hour*(10.0-30)/(10-1e-14) * (m-1e-14))

# initiate classes
hydro   = SHAKTI(mesh, results_dir)
coupler = Coupler_Hydro(mesh, hydro)
coupler.A.assign(2.4e-24)

# geometry
def surface(x,y):
    return 6*((x+5000)**(1/2) - 5000**(1/2)) + 390
def bed(x,y):
    return 350
B = df.Function(coupler.Q_cg).interpolate(bed(x,y))
H = df.Function(coupler.Q_cg).interpolate(surface(x,y)-bed(x,y))

# runoff function
def temp(t):
    return -10*df.cos(2*df.pi*t)- 2

def runoff(t):
    lr    = -0.005
    DDF   = 0.01*365    # m / a / °C
    zs    = surface(x,y)
    return df.max_value(0, (zs*lr+temp(t))*DDF)


# set geometries and variables
coupler.set_geometry(B, H)
hydro.set_coupler(coupler)
hydro.build_variables()
hydro.build_forms(m=0.05, dt0=dt0, e_v=1e-4, h_r=0.5, l_r=10, u_b=30, omega=1/2000)  # 1e-6*s_per_day*365
hydro.write_variables_pvd(0)

# for initial state, take steady state solution from a different run
# chk_file    = results_dir + "initial_fields_A4.h5"
# with CheckpointFile(chk_file, 'r') as afile:
#     mesh_ = afile.load_mesh()
#     hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
#     hydro.set_initial_h(afile.load_function(mesh_, "h"))

# solver options
par = {"snes_type": "newtonls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e0,
       "snes_max_it": 100,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

# time stepping and solve
t     = 0.0
t_end = 15
d     = 0
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    # print("Time = {:.2f} years, dt = {:.1f} days".format(t / (3600*24*365), dt / s_per_day))
    print("Time = {:.2f} years, dt = {:.3f} days".format(t, dt*365))
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=par)
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
        if int(t*365) > d+10:
            d = int(t*365)
            hydro.write_variables_pvd(t)

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# save end result such that it can serve as initial field for another simulation
# if shmip_suit == "A1":
chk_file_save = results_dir + "initial_fields_hill_geom.h5"
hydro.save_end_state(chk_file_save)
