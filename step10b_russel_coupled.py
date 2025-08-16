import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
from models_main.coupled_model import GLADS, SpecFO, Coupler
import models_main.helpers as hlp
import numpy as np
import pandas as pd

import rasterio as rio
from firedrake.__future__ import interpolate

s_per_day = 3600 * 24
results_dir = 'step_10b/results/'
data_dir    = 'step_10b/data/'

m          = 1.158e-9
e_v        = 1e-3

# mesh
# mesh_file = data_dir+'western_med_v2.msh'
mesh_file = 'step_8/data/russel.msh'
mesh = df.Mesh(mesh_file)

# time stepping
dt0 = 0.01/365
dt_max = 12/365
dt_min = 1e-3/365
timestep_increase_fraction = 1.01
timestep_reduction_fraction = 0.5

# geometry
V = df.FunctionSpace(mesh, "CG", 1)
v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
X = df.assemble(interpolate(mesh.coordinates,v_dg))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]
B = df.Function(V)
H = df.Function(V)
with rio.open(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:bed") as src:
    B.dat.data[:] = np.array([pnt[0] for pnt in src.sample(zip(meshx, meshy))])
with rio.open(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:thickness") as src:
    H.dat.data[:] = np.array([pnt[0] for pnt in src.sample(zip(meshx, meshy))])
chk_file    = 'step_10b/data/fenics_geom.h5'
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    # V_ = df.FunctionSpace(mesh_, "CG", 1)
    # B_ = df.Function(V_).interpolate(afile.load_function(mesh_, "B"))
    # H_ = df.Function(V_).interpolate(afile.load_function(mesh_, "H"))
    B.interpolate(afile.load_function(mesh_, "B"))
    H.interpolate(afile.load_function(mesh_, "H"))

# df.project(B_, B)
# df.project(H_, H)


# set subdomain id to 1 where hydro dirichlet bcs should be applied applied
# Vdiv           = df.FunctionSpace(mesh, "HDiv Trace", 0)   # trace elements
# exterior_nodes = Vdiv.boundary_nodes('on_boundary')
# H_div          = df.Function(Vdiv).interpolate(H)          # ice thickness on trace elements
# B_div          = df.Function(Vdiv).interpolate(B)
# edgefunc       = df.Function(Vdiv)                         # function that will hold 1 where bc should be applied, 0 elsewhere
# edgefunc.vector()[exterior_nodes] = (H_div.vector()[exterior_nodes] < 80) # * (B_div.vector()[exterior_nodes] < 200)
# mesh     = df.RelabeledMesh(mesh, [edgefunc], [1])

# set minimum ice thickness to 10
thklim = 0
thklim = 10
Htemp = H.vector().get_local()
Htemp[Htemp<thklim] = thklim
# Htemp[np.isnan(Htemp)] = thklim
H.vector().set_local(Htemp)

# initiate classes with updated mesh
hydro   = GLADS(mesh, results_dir)
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler(mesh, stokes, hydro)

# Convenience functions for calculating the N scale
ones = df.Function(coupler.Q_cg)
ones.vector()[:] = 1.
area = df.assemble(ones*df.dx)
H_mean = df.assemble(H*df.dx)/area
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

f_B = VTKFile(results_dir+"B.pvd").write(B)
f_H = VTKFile(results_dir+"H.pvd").write(H)
f_N = VTKFile(results_dir+"N.pvd")
N   = df.Function(coupler.Q_cg)

# set geometries and variables
coupler.set_geometry(B, H)
stokes.set_coupler(coupler)
hydro.set_coupler(coupler)
hydro.build_variables()
stokes.build_variables()
hydro.build_forms(m, dt0=dt0, e_v=e_v) #, h_r=0.1, k_c=1e-1, k_s=5e-3, l_c=2.0)
stokes.build_forms(beta2=1e6, q=1.0, p=1.0, Nhat=Nhat, Uhat=Uhat)
hlp.plot_geometry(coupler.B, coupler.H, mesh)

x, y = df.SpatialCoordinate(mesh)
hydro.set_initial_S(0.001)
# hydro.set_initial_S(1*(10-(x+2.3e5)/3e5))
# hydro.set_initial_phi(0.0)

solver_params = {#"snes_linesearch_type": "l2",#newton
                 "snes_type":"newtonls",
                 "pc_factor_mat_solver_type": "mumps", # ?
                 "snes_rtol": 1e-3,
                 "snes_atol": 1e0,
                 "snes_max_it": 100,
                 "report": True,
                 "snes_monitor": None,
                 "error_on_nonconvergence": True}

# time stepping and solve
t     = 0.0
d     = 0    # count the days
t_end = 6
while (t <= t_end):
    dt = float(hydro.dt.values()[0])
    # dt = t_end
    print(f"Time = {t} years, dt = {dt*365} days")
    if dt < dt_min:
        print("Minimal time step reached. Simulation failed.")
        break
    try:

        df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=solver_params)
        # df.solve(coupler.R == 0, coupler.U, solver_parameters=solver_params)
        f_N.write(N.interpolate(hydro.N))
        hydro.update_time_variables()
        t += dt
        hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
        hydro.write_variables_pvd(t)
        stokes.write_variables_pvd(t)
        # if int(t*365) >= d+10:
        #     d = int(t*365)
        #     print(d)
        #     hydro.write_variables_pvd(t)
        #     stokes.write_variables_pvd(t)

    except df.exceptions.ConvergenceError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(dt*timestep_reduction_fraction)
        print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
hlp.scatterplt_fields(coupler.U.subfunctions[4:], ["phi", "h", "S"], df.MixedElement(hydro.elements), mesh, results_dir, "russel")
# hlp.scatterplt_fields(coupler.U.subfunctions[0:2], ["ubar_x", "ubar_y"], df.MixedElement(stokes.elements[0:2]), mesh, results_dir, "russel")
