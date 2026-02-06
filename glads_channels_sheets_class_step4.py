import firedrake as df
from GlaDS_main.hydro_class import GLADS
import GlaDS_main.helpers as hlp
from firedrake.checkpointing import CheckpointFile
from firedrake.__future__ import interpolate
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, triplot


s_per_day = 3600 * 24
results_dir = 'step_4/'

# mesh
nx, ny = 75, 25
Lx, Ly = 100e3, 20e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)

# shmip
shmip_suit = "A1"
shmip_m = {"A1" : 7.93e-11,
           "A2" : 1.59e-9,
           "A3" : 5.79e-9,
           "A4" : 2.5e-8,
           "A5" : 4.5e-8,
           "A6" : 5.79e-7}
m = shmip_m[shmip_suit]
shmip_suit = "B1"
# for B4 and B5, it works to start from A1, with dtmax=10 and timestep_increase_fraction = 1.1
# B1-B3: start from B4, dtmax=5 and timestep_increase_fraction=1.05

# geometry
def surface(x,y):
    return 6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1
def bed(x,y):
    return 0

# time stepping
dt0 = s_per_day*0.01
dt_max = s_per_day*5
dt_min = s_per_day*1e-5
timestep_increase_fraction = 1.05
timestep_reduction_fraction = 0.5

# read in location of moulins
df_moul = pd.read_csv(f'/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/SHMIP_results/{shmip_suit}_M.csv', names=["idx","x","y","m"])

# use closest nodes instead of exactly the coordinates in df_moul
v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
X = df.assemble(interpolate(mesh.coordinates,v_dg))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]

xx = np.zeros(len(df_moul.x))
yy = np.zeros(len(df_moul.y))
i  = 0
for (mx,my,Q_in) in zip(df_moul.x,df_moul.y,df_moul.m):
    ii = np.argmin(np.sqrt((meshx-mx)**2+(meshy-my)**2))
    xx[i] = meshx[ii]
    yy[i] = meshy[ii]
    i += 1
df_moul2 = pd.DataFrame({"x":xx, "y":yy, "m":df_moul.m})

# hydro object
hydro = GLADS(mesh, results_dir)
hydro.build_variables(m, df_moul2, dt0, surface, bed)
hlp.plot_geometry(hydro)


# for initial state, take steady state solution from a different run
chk_file    = results_dir + "initial_fields_B4.h5"
csv_file    = results_dir + "initial_S_B4.csv"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))
# hydro.set_initial_S(1.0)
# x, y = df.SpatialCoordinate(hydro.mesh)
# hydro.set_initial_S(20*(1-x/100e3))

# solver options
par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e-5,
       "snes_max_it": 100,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

# time stepping and solve
t     = 0.0
t_end = s_per_day*365*25
d     = 0
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    print([t / (3600*24*365), dt / s_per_day])
    # print([t, dt*365])
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:
        df.solve(hydro.F == 0, hydro.U, bcs=hydro.bcs, solver_parameters=par)
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
        if int(t/(s_per_day)) >= d+10:
                d = int(t/(s_per_day))
                hydro.write_variables_pvd(d/365)

        # Doug's trick to save Q
        # df.solve(F_Q == 0, dQ)
        # outfile_Q.write(df.project(dQ, V_S, name="Q"))

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
hlp.scatterplt_fields(hydro, shmip_suit)

# # test against GlaDS-matlab SHMIP results
# fl = "SHMIP_results/mw/"+shmip_suit+"_mw.nc"
# hlp.diff_to_glads_matlab(hydro, fl)
