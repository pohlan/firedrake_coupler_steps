import firedrake as df
from models_main.coupled_model import SHAKTI, GLADS, Coupler_Hydro
# from models_main.hydro_class import SHAKTI
import models_main.helpers as hlp
from firedrake.checkpointing import CheckpointFile
import numpy as np
import pandas as pd

s_per_day = 3600 * 24
results_dir = 'step_11b/'

# mesh
nx, ny = 75, 25
Lx, Ly = 100e3, 25e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
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

# get moulin coordinates
df_moul = pd.read_csv(f'/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/SHMIP_results/moulins_68_Hill.csv', names=["idx","x","y"])
coords = hlp.get_coordinates(mesh, "CG", 1)
meshx = coords[:,0]
meshy = coords[:,1]
x_moulin = np.zeros(len(df_moul.x))
y_moulin = np.zeros(len(df_moul.y))
id_mesh = np.zeros(len(df_moul.y), dtype=int)
i  = 0
for (mx,my) in zip(df_moul.x,df_moul.y):
    ii = np.argmin(np.sqrt((meshx-mx)**2+(meshy-my)**2))
    x_moulin[i] = meshx[ii]
    y_moulin[i] = meshy[ii]
    id_mesh[i] = ii
    i += 1

# get center coordinates of cells (DG0 DoFs)
DG0 = df.FunctionSpace(mesh, "DG", 0)
x_cell = df.Function(DG0).interpolate(x)
y_cell = df.Function(DG0).interpolate(y)
# make subdomain for each moulin catchment for RelabeledMesh
facet_function = df.Function(DG0).interpolate(-1)
facet_functions = [df.Function(DG0) for i in np.arange(len(df_moul.x))]
for (msh_i,(msh_x,msh_y)) in enumerate(zip(x_cell.dat.data_ro, y_cell.dat.data_ro)):
    i_downstream = np.where(x_moulin < msh_x + 1)
    if i_downstream[0].size == 0:
        continue
    i_min = np.argmin(np.sqrt((x_moulin[i_downstream]-msh_x)**2+(y_moulin[i_downstream]-msh_y)**2))
    facet_functions[i_downstream[0][i_min]].dat.data[msh_i] = 1
mesh_c = df.RelabeledMesh(mesh, facet_functions, list(range(len(facet_functions))))
dx_c   = df.dx(domain=mesh_c)

# set geometries and variables
coupler.set_geometry(B, H)
hydro.set_coupler(coupler)
hydro.build_variables()
hydro.build_forms(m=0.05, dt0=dt0, e_v=1e-4, h_r=0.5, l_r=10, u_b=30, omega=1/2000, moulins=True)
hydro.write_variables_pvd(0)

# initial melt input to moulins
DG0 = df.FunctionSpace(mesh_c, "DG", 0)
m_input = df.Function(DG0).interpolate(runoff(0))
m_file = df.VTKFile(results_dir+"m_input.pvd")
if hydro.moulins:
    M_moulins = np.zeros(len(df_moul.x))
    for i in range(len(M_moulins)):
        M_moulins[i] = df.assemble(m_input*dx_c(i))
    Qm, delta_moul = hlp.moulin_dirac_from_array(mesh, x_moulin, y_moulin, M_moulins)
    hydro.Qm.interpolate(Qm)
    hydro.delta_moul.interpolate(delta_moul)
    Qm_save = df.Function(hydro.V_phi)
    Qm_file = df.VTKFile(results_dir+"moulin_dirac.pvd")

# for initial state, take steady state solution from a different run
chk_file    = results_dir + "initial_fields_hill_geom.h5"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))

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
t     = 0
t_end = 2
d     = 0
with df.CheckpointFile(f"{results_dir}time_series.h5", 'w') as afile:
    afile.save_mesh(mesh)
    i     = 1    # idx for checkpointing
    while (t <= t_end):
        dt = float(hydro.dt.values()[0])
        # print("Time = {:.2f} years, dt = {:.3f} days".format(t, dt*365))
        print("Time = {:.2f} years, dt = {:.1f} hours".format(t, dt*365*24))
        if dt < dt_min:
            print("Minimal time step reached. Simulation failed.")
            break
        try:
            m_input.interpolate(runoff(t))
            m_file.write(m_input, time=t)

            if hydro.moulins:
                for ii in range(len(M_moulins)):
                    M_moulins[ii] = df.assemble(m_input*dx_c(ii))
                Qm, delta_moul = hlp.moulin_dirac_from_array(mesh, x_moulin, y_moulin, M_moulins)
                hydro.Qm.interpolate(Qm)
                Qm_save.interpolate(Qm)
                Qm_file.write(Qm_save, time=t)
            else:
                hydro.m.interpolate(m_input)

            df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=par)
            hydro.update_time_variables()
            t += dt
            dt_max = get_dt(np.max(m_input.dat.data_ro))
            hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
            if int(t*365) > d+10:
                d = int(t*365)
                hydro.write_variables_pvd(t)
                afile.save_function(df.project(hydro.P_w/(coupler.rho_i*coupler.g*coupler.H),coupler.Q_cg), idx=i, name="pw_pi")
                i += 1

        except df.exceptions.ConvergenceError:
            # If solver fails, try again with a smaller time step
            hydro.dt.assign(dt*timestep_reduction_fraction)
            print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# save end result such that it can serve as initial field for another simulation
# if shmip_suit == "A1":
# chk_file_save = results_dir + "initial_fields_hill_geom.h5"
# hydro.save_end_state(chk_file_save)
