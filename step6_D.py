import firedrake as df
from GlaDS_main.hydro_class import GLADS
import GlaDS_main.helpers as hlp
from firedrake.checkpointing import CheckpointFile
import numpy as np
import pandas as pd

s_per_hour  = 3600
s_per_day   = s_per_hour * 24
s_per_year  = s_per_day * 365
results_dir = 'step_6/'

# mesh
nx, ny = 45, 15
Lx, Ly = 100e3, 20e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
x, y = df.SpatialCoordinate(mesh)

# shmip
shmip_suit = "D5"
shmip_DT = {"D1" : -4,
            "D2" : -2,
            "D3" :  0,
            "D4" :  2,
            "D5" :  4}
DT = shmip_DT[shmip_suit]

# geometry
def surface(x,y):
    return 6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1
def bed(x,y):
    return 0

def temp(t):
    return -16*df.cos(2*df.pi/s_per_year*t)- 5 + DT
def runoff(t):
    lr    = -0.0075
    DDF   = 0.01/86400
    basal = 7.93e-11
    zs    = surface(x,y)
    return df.max_value(0, (zs*lr+temp(t))*DDF) + basal
m = runoff(0.0)
# print(m)

# time stepping
dt0 = s_per_hour*1
dt_max = s_per_hour*5
dt_min = s_per_hour*1e-2
timestep_increase_fraction = 1.1
timestep_reduction_fraction = 0.5
def get_dt(m):
    return max(5.0*s_per_hour, s_per_hour*15 + s_per_hour*(0.5-15)/(8.1e-7-7.93e-11) * (m-7.93e-11))

# hydro object
hydro = GLADS(mesh, results_dir)
hydro.build_variables(m, dt0, surface(x,y)-bed(x,y), bed(x,y))

hlp.plot_geometry(hydro)


# for initial state, take steady state solution from a different run
chk_file    =  "step_4/initial_fields_A1.h5"
csv_file    =  "step_4/initial_S_A1.csv"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))

# solver options
par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e-5,
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
        # hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
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

# save end result such that it can serve as initial field for another simulation
# if shmip_suit == "A1":
# chk_file_save = results_dir + "initial_fields_"+shmip_suit+".h5"
# csv_file_save = results_dir + "initial_S_"+shmip_suit+".csv"
# hydro.save_end_state(chk_file_save, csv_file_save)

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
# hlp.scatterplt_fields(hydro, shmip_suit)

# test against GlaDS-matlab SHMIP results
# fl = "SHMIP_results/mw/"+shmip_suit+"_mw.nc"
# hlp.diff_to_glads_matlab(hydro, fl)
