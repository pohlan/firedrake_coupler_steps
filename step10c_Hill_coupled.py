import firedrake as df
from firedrake.checkpointing import CheckpointFile
from models_main.coupled_model import GLADS, Coupler
import models_main.helpers as hlp
import numpy as np
import pandas as pd

s_per_day = 3600 * 24
s_per_year = s_per_day*365
results_dir = 'step_10c_trans_32/'

# mesh
# nx, ny = 125, 40
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
hydro   = GLADS(mesh, results_dir)
coupler = Coupler(mesh, hydro)
coupler.A.assign(2.4e-24) # a bit lower than e.g. in SHMIP

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

# import matplotlib.pyplot as plt
# def runoff_plt(t):
#     T     = -10*np.cos(2*np.pi*t)- 2
#     lr    = -0.005
#     DDF   = 0.01*365    # m / a / °C
#     basal = 0.05
#     zs    = 390 # surface(x,y)
#     return np.maximum((zs*lr+T)*DDF, 0) # + basal
# tt = np.arange(0,1,0.005)
# plt.plot(tt,runoff_plt(tt)/365)
# plt.savefig("runoff.jpg")

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
hydro.build_forms(m=0.05, dt0=dt0, e_v=1e-4, k_s=0.05, k_c=0.5, h_r=0.5, l_r=10, l_c=10, transition=True, alpha_s=1.5, u_b=30, moulins=True)
hlp.plot_geometry(coupler.B, coupler.H, mesh)

# initial melt input to moulins
DG0 = df.FunctionSpace(mesh_c, "DG", 0)
m_input = df.Function(DG0).interpolate(runoff(0))
M_moulins = np.zeros(len(df_moul.x))
for i in range(len(M_moulins)):
    M_moulins[i] = df.assemble(m_input*dx_c(i))
Qm, delta_moul = hlp.moulin_dirac_from_array(mesh, x_moulin, y_moulin, M_moulins)
hydro.Qm.interpolate(Qm)
hydro.delta_moul.interpolate(delta_moul)
Qm_save = df.Function(hydro.V_phi)
Qm_file = df.VTKFile(results_dir+"moulin_dirac.pvd")
m_file = df.VTKFile(results_dir+"m_input.pvd")

# initialize fields
chk_file = "step_10c_trans_54/initial_fields_hill.h5"
csv_file = "step_10c_trans_54/initial_S_hill.csv"
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))

# solver parameters
solver_params = {"snes_type": "newtonls",#newton
                 "pc_factor_mat_solver_type": "mumps", # ?
                 "snes_rtol": 1e-5,
                 "snes_atol": 1e0,
                 "snes_max_it": 100,
                 "report": True,
                 "snes_monitor": None,
                 "error_on_nonconvergence": True}

# time stepping and solve
t     = 0
t_end = 2
d     = 0  # count the days
with df.CheckpointFile(f"{results_dir}time_series.h5", 'w') as afile:
    afile.save_mesh(mesh)
    i     = 1    # idx for checkpointing

    while (t <= t_end):
        dt = float(hydro.dt.values()[0])
        # print(f"Time = {t} years, dt = {dt*365} days")
        print("Time = {:.2f} years, dt = {:.1f} hours".format(t, dt*365*24))
        if dt < dt_min:
            print("Minimal time step reached. Simulation failed.")
            break
        try:
            m_input.interpolate(runoff(t))

            for ii in range(len(M_moulins)):
                M_moulins[ii] = df.assemble(m_input*dx_c(ii))

            Qm, delta_moul = hlp.moulin_dirac_from_array(mesh, x_moulin, y_moulin, M_moulins)
            hydro.Qm.interpolate(Qm)
            Qm_save.interpolate(Qm)
            Qm_file.write(Qm_save, time=t)
            m_file.write(m_input, time=t)

            df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=solver_params)
            hydro.update_time_variables()
            t += dt
            dt_max = get_dt(np.max(m_input.dat.data_ro))
            hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
            if int(t*365) >= d+2:
                d = int(t*365)
                hydro.write_variables_pvd(t)
                afile.save_function(df.project(hydro.P_w/(coupler.rho_i*coupler.g*coupler.H),coupler.Q_cg), idx=i, name="pw_pi")
                i += 1

        except df.exceptions.ConvergenceError:
            # If solver fails, try again with a smaller time step
            hydro.dt.assign(dt*timestep_reduction_fraction)
            print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

# make matplotlib scatterplot for quick visualization (for channels only way of visualizing currently)
# hlp.scatterplt_fields(coupler.U.subfunctions[4:], ["phi", "h", "S"], df.MixedElement(hydro.elements), mesh, results_dir, shmip_suit)
# hlp.scatterplt_fields(coupler.U.subfunctions[0:2], ["ubar_x, ubar_y"], df.MixedElement(stokes.elements[0:2]), mesh, results_dir, shmip_suit)

# chk_file_save = results_dir + "initial_fields_hill.h5"
# csv_file_save = results_dir + "initial_S_hill.csv"
# hydro.save_end_state(chk_file_save, csv_file_save)
