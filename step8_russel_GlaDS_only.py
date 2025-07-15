import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
from models_main.hydro_class import GLADS
import models_main.helpers as hlp
import numpy as np
import pandas as pd

import rasterio as rio
from firedrake.__future__ import interpolate

s_per_day = 3600 * 24
results_dir = 'step_8/'
data_dir    = 'step_8/data/'

m          = 1e-8
e_v        = 1e-3

# mesh
mesh = df.Mesh("russel.msh")

# time stepping
dt0 = s_per_day*0.01
dt_max = s_per_day*45
dt_min = s_per_day*1e-3
timestep_increase_fraction = 1.01
timestep_reduction_fraction = 0.9

# hydro
hydro = GLADS(mesh, results_dir)

# geometry
v_dg = df.VectorFunctionSpace(mesh, hydro.E_V.sub_elements[0])
X = df.assemble(interpolate(mesh.coordinates,v_dg))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]
B = df.Function(hydro.V_phi)
H = df.Function(hydro.V_phi)
with rio.open(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:bed") as src:
    B.dat.data[:] = np.array([pnt[0] for pnt in src.sample(zip(meshx, meshy))])
with rio.open(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:thickness") as src:
    H.dat.data[:] = np.array([pnt[0] for pnt in src.sample(zip(meshx, meshy))])

hydro.build_variables(m, dt0, H, B, e_v)
hlp.plot_geometry(hydro.B, hydro.H, mesh)

par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps",
       "snes_rtol": 1e-5,
       "snes_atol": 1e-4,
       "snes_max_it": 300,
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

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
hlp.scatterplt_fields(hydro, "russel")
