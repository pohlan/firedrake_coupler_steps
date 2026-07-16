import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
from firedrake.checkpointing import CheckpointFile
from models_main.coupled_model import GLADS, SpecFO, Coupler_Flow_Hydro
import models_main.helpers as hlp
from models_main.russel_catchments import get_catchments_russel
from models_main.initialize import initialize
import numpy as np
import pandas as pd
import geoutils as gu

s_per_hour = 3600
s_per_day  = s_per_hour * 24

args = hlp.get_args()   # command line arguments

data_dir    = args.data_directory
results_dir = args.results_directory+"/run_{}/".format(args.run_index)
params_output_file = args.results_directory+"parameter_runs.csv"

# mesh
mesh_file = data_dir+'russel/russel.msh'
mesh = df.Mesh(mesh_file)

# time stepping
dt_max = 20/365
dt_min = 1e-3/365
timestep_increase_fraction = 1.1
timestep_reduction_fraction = 0.5
day = 1/365
hour = day/24
def get_dt(m):
    return max(2.0*hour, 10*hour + hour*(2.0-10)/(10-1e-14) * (m-1e-14))

# geometry
V = df.FunctionSpace(mesh, "CG", 1)
v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
X = df.assemble(df.interpolate(mesh.coordinates,v_dg))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]
B = df.Function(V)
H = df.Function(V)

# load bed and thickness data, either original BedMachine or smoothed
sig = args.sig_topo
if sig==0:
    r_bed = gu.Raster(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:bed")
    r_thk = gu.Raster(f"NETCDF:{data_dir}BedMachineGreenland-v5.nc:thickness")
else:  # higher sigma == more smoothing
    r_bed = gu.Raster(f"{data_dir}BedMachineGreenland-v5_bed_smooth_sig{sig}.nc")
    r_thk = gu.Raster(f"{data_dir}BedMachineGreenland-v5_thickness_smooth_sig{sig}.nc")
# interpolate onto mesh
B.dat.data[:] = r_bed.interp_points((meshx, meshy), as_array=True)
H.dat.data[:] = r_thk.interp_points((meshx, meshy), as_array=True)
S = B.dat.data[:] + H.dat.data[:] # surface elevation

# set minimum ice thickness to 10
thklim = 0
thklim = 10
Htemp = H.dat.data[:]
Htemp[Htemp<thklim] = thklim
H.dat.data[:] = Htemp

# make bed elevation and thickness the same at bc points of individual outlets
bc_nodes = V.boundary_nodes(1)
for i in range(0,len(bc_nodes),2):
    nodes = bc_nodes[i:i+2]
    B.dat.data[nodes] = np.mean(B.dat.data_ro[nodes])
    H.dat.data[nodes] = np.mean(H.dat.data_ro[nodes])

# save geometries
f_B = df.VTKFile(results_dir+"B.pvd").write(B)
f_H = df.VTKFile(results_dir+"H.pvd").write(H)

# initiate classes
hydro   = GLADS(mesh, results_dir)
stokes  = SpecFO(mesh, results_dir)
coupler = Coupler_Flow_Hydro(mesh, stokes, hydro)

# Convenience functions for calculating the N scale
ones = df.Function(coupler.Q_cg)
ones.dat.data[:] = 1.
area = df.assemble(ones*df.dx)
H_mean = df.assemble(H*df.dx)/area
Uhat   = df.Constant(50)
Nhat   = df.Constant(917*9.81*H_mean)

# melt input to hydro model
f_melt = df.VTKFile(results_dir+"m0_per_year.pvd")
if args.melt_input == "KAN":
    m, calc_m = hlp.melt_fct_KAN(hydro, S, data_dir)
elif args.melt_input == "MAR":
    m, calc_m = hlp.melt_fct_MAR_monthly_files(hydro, H, meshx, meshy, coupler)
elif args.melt_input == "avg":
    m, calc_m = hlp.melt_fct_avg(hydro, H)

# moulins
df_moul, facet_functions, distributed_melt_mask = get_catchments_russel(mesh)

# get moulin coordinates that are exactly on the nodes
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

# make subdomain for each moulin catchment for RelabeledMesh
mesh_c = df.RelabeledMesh(mesh, facet_functions, list(range(len(facet_functions))))
dx_c   = df.dx(domain=mesh_c)

# set geometries and variables
coupler.set_geometry(B, H)
hlp.plot_geometry(coupler.B, coupler.H, mesh)
stokes.set_coupler(coupler)
hydro.set_coupler(coupler)
hydro.build_variables()
stokes.build_variables()
# dt0=get_dt(max(melt[:,0]))
dt0= 2*hour

# get beta2 from inversion
# chk_file = "test_inversion/beta2_opt_1e-3.h5"
chk_file = "test_inversion/beta2_opt_1e-2.5_run311_new.h5"
with CheckpointFile(chk_file, 'r') as afile:
        mesh_ = afile.load_mesh()
        beta2 = afile.load_function(mesh_, name="beta2")
df.VTKFile(results_dir+"beta2.pvd").write(beta2)

stokes.build_forms(beta2=beta2, q=args.q, p=args.p, Nhat=Nhat, Uhat=Uhat)
# stokes.build_forms(beta2=args.beta2, q=args.q, p=args.p, Nhat=Nhat, Uhat=Uhat)
hydro.build_forms(m, dt0=dt0, e_v=args.e_v, h_r=args.h_r, k_c=args.k_c, k_s=args.k_s, l_c=args.l_c, l_r=args.l_r, transition=args.transition, alpha_s=args.alpha_s, beta_s=args.beta_s, omega=args.omega, As_factor=args.As_factor, moulins=args.moulins)

# initial melt input to moulins
calc_m(0)
DG0 = df.FunctionSpace(mesh_c, "DG", 0)
m_DG0 = df.Function(DG0).interpolate(hydro.m)
m_CG1 = df.Function(coupler.Q_cg)
# m_file = df.VTKFile(results_dir+"m_input.pvd")
if hydro.moulins:
    M_moulins = np.zeros(len(df_moul.x))
    for i in range(len(M_moulins)):
        M_moulins[i] = df.assemble(m_DG0*dx_c(i))
    Qm, delta_moul = hlp.moulin_dirac_from_array(mesh, x_moulin, y_moulin, M_moulins)
    hydro.Qm.interpolate(Qm)
    hydro.delta_moul.interpolate(delta_moul)
    # Qm_save = df.Function(hydro.V_phi)
    Qm_file = df.VTKFile(results_dir+"moulin_dirac.pvd")

# initialize (run into steady state)
chk_file, csv_file = initialize(mesh, H, B, Uhat, Nhat, args, beta2, results_dir)
# chk_file, csv_file = initialize(mesh, H, B, Uhat, Nhat, args, args.beta2, results_dir)
with CheckpointFile(chk_file, 'r') as afile:
    mesh_ = afile.load_mesh()
    hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
    hydro.set_initial_h(afile.load_function(mesh_, "h"))
hydro.set_initial_S(np.float64(pd.read_csv(csv_file).S))

solver_params = {#"snes_linesearch_type": "l2",#newton
                 "snes_type":"newtonls",
                 "pc_factor_mat_solver_type": "mumps", # ?
                 "snes_rtol": 1e-3,
                 "snes_atol": 1e-3,
                 "snes_max_it": 50,
                 "report": False,
                #  "snes_monitor": None,
                 "error_on_nonconvergence": True}

# time stepping and solve
t       = 0.0
d       = 0    # count the days
t_end   = args.t_end
success = True
with df.CheckpointFile(f"{results_dir}/time_series.h5", 'w') as afile:
    afile.save_mesh(mesh)
    i     = 1    # idx for checkpointing

    while (t <= t_end):
        dt = float(hydro.dt.values()[0])
        print(np.max(m_CG1.dat.data_ro))
        print("Time = {:.2f} years, dt = {:.1f} hours".format(t, dt*365*24))
        if dt < dt_min:
            # write failure to table
            success = False
            print("Minimal time step reached. Simulation failed.")
            break
        try:
            # interpolate spatially resolved surface runoff
            m_CG1.dat.data[:] = calc_m(t) + args.m_basal

            if hydro.moulins:
                m_DG0.interpolate(m_CG1)
                # distribute m_input to moulins
                for ii in range(len(M_moulins)):
                    M_moulins[ii] = df.assemble(m_DG0*dx_c(ii))
                Qm, delta_moul = hlp.moulin_dirac_from_array(mesh, x_moulin, y_moulin, M_moulins)
                hydro.Qm.interpolate(Qm)
                hydro.m.interpolate(m_CG1*distributed_melt_mask)
            else:
                hydro.m.interpolate(m_CG1)

            # Downs et al variable sheet conductivity
            # kmin = df.Constant(1e-3*s_per_day*365)
            # kmax = df.Constant(1e-2*s_per_day*365)
            # hydro.k_s.interpolate((kmax-kmin)/25 * hydro.m + kmin)

            # for w, form in zip([hydro.w_phi, hydro.w_h, hydro.w_S, stokes.w_R_u, stokes.w_R_v], [hydro.R_phi_h, hydro.R_h, hydro.R_S, stokes.R_u_body, stokes.R_v_body]):
            #     # print(1/df.assemble(form).dat.norm)
            #     w.assign(1/df.assemble(form).dat.norm)

            # terms = {
            #         # "phi_storage": hydro.T_storage,
            #         # "phi_flux": hydro.T_flux,
            #         # "phi_source": hydro.T_source,
            #         # "h_time": hydro.h_time,
            #         # "h_O": hydro.h_O,
            #         # "h_C": hydro.h_C,
            #         # "S_time": hydro.S_time,
            #         # "S_O": hydro.S_O,
            #         # "S_C": hydro.S_C,
            #         # "S_boundary": hydro.S_bc,
            #         "R_phi_h": hydro.R_phi_h_print,
            #         "R_phi_S": hydro.R_phi_S_print,
            #         "R_h": hydro.R_h_print,
            #         "R_S": hydro.R_S_print,
            #         "R_u_body": stokes.R_u_body_print,
            #         "R_v_body": stokes.R_v_body_print
            #         }
            # for name, form in terms.items():
            #     print(f"{name:15s}: {df.assemble(form).dat.norm:.3e}")

            df.solve(coupler.R == 0, coupler.U, bcs=hydro.bcs, solver_parameters=solver_params)
            hydro.update_time_variables()
            # stokes.u0 = stokes.u
            # stokes.v0 = stokes.v
            t += dt
            dt_max = get_dt(np.max(m_CG1.dat.data_ro))
            hydro.dt.assign(min(dt*timestep_increase_fraction,dt_max))
            if int(t*365) >= d+2:
                d = int(t*365)
                f_melt.write(m_CG1, time=t)
                # hydro.write_variables_pvd(t)
                stokes.write_variables_pvd(t)
                afile.save_function(stokes.Us, idx=i, name="Us")
                # afile.save_function(stokes.Ub, idx=i, name="Ub")
                afile.save_function(coupler.U.sub(0), idx=i, name="phi")
                afile.save_function(df.project(hydro.q_s, coupler.Q_cg_vec), idx=i, name="q")
                Q_subDG0 = hlp.save_DGT0(mesh, hydro.Q_submesh, df.project(abs(hydro.Q),hydro.V_S), hydro.Q_save, hydro.outfile_Q, t)
                afile.save_function(Q_subDG0, idx=i, name="Q")
                afile.save_function(coupler.U.sub(1), idx=i, name="h")
                afile.save_function(m_CG1, idx=i, name="m")
                afile.save_function(df.project(hydro.N, coupler.Q_cg), idx=i, name="N")
                i += 1

        except df.exceptions.ConvergenceError:
            # If solver fails, try again with a smaller time step
            coupler.U.sub(0).assign(hydro.phi0)
            coupler.U.sub(1).assign(hydro.h0)
            coupler.U.sub(2).assign(hydro.S0)
            hydro.dt.assign(dt*timestep_reduction_fraction)
            print("Convergence not achieved.  Reducing time step to {:.1f} hours and trying again".format(hydro.dt.values()[0]*365*24))

# write parameters to table:
hlp.save_params_to_csv(args, params_output_file, success=success)

# chk_file = results_dir+"winter_state.h5"
# csv_file = results_dir+"winter_state.csv"
# hydro.save_end_state(chk_file, csv_file)
