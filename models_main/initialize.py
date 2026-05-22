import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
from firedrake.checkpointing import CheckpointFile
from models_main.coupled_model import GLADS, SpecFO, Coupler_Flow_Hydro, Coupler_Hydro
import models_main.helpers as hlp
import numpy as np
import pandas as pd

s_per_day = 3600 * 24
results_dir = 'models_main/initial_states/'

# constant melt input (m/yr)
m          = 0.003

# time stepping
dt0 = 0.05/365
dt_max = 45/365
dt_min = 1e-3/365
timestep_increase_fraction = 1.05
timestep_reduction_fraction = 0.5

def initialize(mesh, H, B, Uhat, Nhat, args):

    # initiate classes
    hydro   = GLADS(mesh, results_dir)
    coupler = Coupler_Hydro(mesh, hydro)

    # set geometries and variables
    coupler.set_geometry(B, H)
    hlp.plot_geometry(coupler.B, coupler.H, mesh)
    hydro.set_coupler(coupler)
    hydro.build_variables()
    hydro.build_forms(u_b=100, m=m, dt0=dt0, e_v=args.e_v, h_r=args.h_r, k_c=args.k_c, k_s=args.k_s, l_c=args.l_c, l_r=args.l_r, transition=args.transition, alpha_s=args.alpha_s, beta_s=args.beta_s, omega=args.omega, As_factor=args.As_factor, moulins=args.moulins)

    solver_params = {#"snes_linesearch_type": "l2",#newton
                     "snes_type":"newtonls",
                     "pc_factor_mat_solver_type": "mumps", # ?
                     "snes_rtol": 1e-3,
                     "snes_atol": 1e0,
                     "snes_max_it": 50,
                     "report": False,
                    #  "snes_monitor": None,
                     "error_on_nonconvergence": True}

    #########################################
    # solve for the hydro-only steady state #
    #########################################

    print("Initializing hydro-only model..")

    # time stepping and solve
    t     = 0.0
    d     = 0    # count the days
    t_end = 20   # in years
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
            # if int(t*365) >= d+10:
            #     d = int(t*365)
            #     hydro.write_variables_pvd(t)

        except df.exceptions.ConvergenceError:
            # If solver fails, try again with a smaller time step
            hydro.dt.assign(dt*timestep_reduction_fraction)
            print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

    # save end states for initialization of coupled run
    chk_file_hydro = results_dir + f"initial_fields_russel_hydro_sig{args.sig_topo}.h5"
    csv_file_hydro = results_dir + f"initial_S_russel_hydro_sig{args.sig_topo}.csv"
    hydro.save_end_state(chk_file_hydro, csv_file_hydro)

    ######################################
    # solve for the coupled steady-state #
    ######################################

    print("Initializing coupled model..")

    # initiate classes
    hydro   = GLADS(mesh, results_dir)
    stokes  = SpecFO(mesh, results_dir)
    coupler = Coupler_Flow_Hydro(mesh, stokes, hydro)

    # set geometries and variables
    coupler.set_geometry(B, H)
    stokes.set_coupler(coupler)
    hydro.set_coupler(coupler)
    hydro.build_variables()
    stokes.build_variables()
    stokes.build_forms(beta2=args.beta2, q=args.q, p=args.p, Nhat=Nhat, Uhat=Uhat)
    hydro.build_forms(m=m, dt0=dt0, e_v=args.e_v, h_r=args.h_r, k_c=args.k_c, k_s=args.k_s, l_c=args.l_c, l_r=args.l_r, transition=args.transition, alpha_s=args.alpha_s, beta_s=args.beta_s, omega=args.omega, As_factor=args.As_factor, moulins=args.moulins)

    # take steady-state from hydro run as initialization
    with CheckpointFile(chk_file_hydro, 'r') as afile:
        mesh_ = afile.load_mesh()
        hydro.set_initial_phi(afile.load_function(mesh_, "phi"))
        hydro.set_initial_h(afile.load_function(mesh_, "h"))
    hydro.set_initial_S(np.float64(pd.read_csv(csv_file_hydro).S))

    # time stepping and solve
    t     = 0.0
    d     = 0    # count the days
    t_end = 20   # in years
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
            # if int(t*365) >= d+10:
            #     d = int(t*365)
            #     hydro.write_variables_pvd(t)
            #     stokes.write_variables_pvd(t)

        except df.exceptions.ConvergenceError:
            # If solver fails, try again with a smaller time step
            hydro.dt.assign(dt*timestep_reduction_fraction)
            print("Convergence not achieved.  Reducing time step to {0} days and trying again".format(hydro.dt.values()[0] / s_per_day))

    # save end states for initialization of seasonal simulations
    chk_file_save = results_dir + f"initial_fields_russel_coupled_sig{args.sig_topo}.h5"
    csv_file_save = results_dir + f"initial_S_russel_coupled_sig{args.sig_topo}.csv"
    hydro.save_end_state(chk_file_save, csv_file_save)

    print("Running time-dependent model..")

    return chk_file_save, csv_file_save
