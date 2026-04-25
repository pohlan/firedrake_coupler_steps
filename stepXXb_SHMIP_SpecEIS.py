import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
# sys.path.insert(0, '/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/GlaDS_main/')
# os.chdir('/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/')
# print(os.getcwd())
import firedrake as df
from firedrake.output import VTKFile
from firedrake.__future__ import interpolate
import pickle
from firedrake.petsc import PETSc
from models_main.SpecEIS import CoupledModel
import rasterio as rio
import numpy as np


import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, triplot


class Glacier:

    def read_geometry(self, H, B):
    #     v_dg = df.VectorFunctionSpace(self.model.mesh, self.model.E_thk)
    #     X = df.assemble(interpolate(self.model.mesh.coordinates,v_dg))
    #     meshx = X.dat.data_ro[:,0] * 108000 # multiplied with len_scale
    #     meshy = X.dat.data_ro[:,1] * 108000 # multiplied with len_scale
    #     with rio.open(f"NETCDF:{nc_path}:bed") as src:
    #         self.model.B.dat.data[:] = np.array([pnt[0] for pnt in src.sample(zip(meshx, meshy))]) / self.model.thk_scale
    #     # with rio.open(f"NETCDF:{nc_path}:thickness") as src:
    #         # self.model.H0.dat.data[:] = np.array([pnt[0] for pnt in src.sample(zip(meshx, meshy))]) / self.model.thk_scale
        self.model.H0.interpolate(H/self.model.thk_scale)
        self.model.B.interpolate(B/self.model.thk_scale)

    # def interpolate_bed_from_pickle(self,interpolant_path):
    #     interpolant = pickle.load(open(interpolant_path,'rb'))

    #     v_dg = df.VectorFunctionSpace(self.model.mesh,self.model.E_thk)
    #     X = df.interpolate(self.model.mesh.coordinates,v_dg)
    #     self.model.B.dat.data[:] = (interpolant(
    #         X.dat.data_ro[:,0],X.dat.data_ro[:,1],grid=False)/1000.0)
    #     self.model.H0.interpolate(df.Constant(0.001))

    def __init__(self,results_dir,data_dir,conservation_test=False,init_dir=None):
        # if conservation_test:
        #     with df.CheckpointFile(f"{init_dir}/functions.h5", 'r') as afile:
        #         mesh = afile.load_mesh('mesh')
        # else:
        #     mesh = df.Mesh(f'{data_dir}russel.msh',name='mesh')
        mesh = df.Mesh("valley.msh")

        config = {'solver_type': 'gmres',
                  'velocity_function_space':'MTW',
                #   'sia' : True,
                  'sliding_law': 'linear',
                  'vel_scale': 100.,
                  'thk_scale': 1000.,
                  'len_scale': 108000.,
                  'beta_scale': 1000.,
                  'theta': 1.0,
                  'thklim': 1e-3,
                  'alpha': 1000.0,
                #   'boundary_markers':[1,2],
                  'z_sea': -0.4,
                #   'calve': True
                  }

        model = self.model = CoupledModel(mesh,**config)

        fig, axes = plt.subplots()
        triplot(mesh, axes=axes)
        axes.legend()
        axes.axis("equal")
        plt.savefig("russel_mesh.jpg")

        # self.interpolate_bed_from_pickle(f'{data_dir}/interpolant.pkl')
        # self.read_geometry(f'{data_dir}BedMachineGreenland-v5.nc')

        # geometry
        shmip_suit = "E1"
        para_bench = 0.05
        shmip_para = {"E1":  0.05,
                      "E2":  0.0 ,
                      "E3": -0.1 ,
                      "E4": -0.5 ,
                      "E5": -0.7 }
        para = shmip_para[shmip_suit]
        def surface(x,y):
            return 100*(x+200)**(1/4) + 1/60*x - 2e10**(1/4) + 1
        def f(x,para):
            return (surface(6e3,0) - para*6e3)/6e3**2 * x**2 + para*x
        def g(y):
            return 0.5e-6 * abs(y)**3
        def h(x,para):
            return (-4.5*x/6e3 + 5) * (surface(x,0)-f(x, para)) / (surface(x,0)-f(x, para_bench)+1e-15)
        def bed(x,y):
            return f(x,para) + g(y) * h(x,para)

        x, y = df.SpatialCoordinate(mesh)
        B = df.Function(model.Q_thk).interpolate(bed(x,y))
        H = df.Function(model.Q_thk).interpolate(surface(x,y)-bed(x,y))
        self.read_geometry(H,B)

        fig, axes = plt.subplots()
        cl = tripcolor(model.H0, axes=axes)
        fig.colorbar(cl)
        plt.savefig("H_f.jpg")

        ####
        # from firedrake.pyplot import tripcolor, triplot
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots()
        # cl = tripcolor(model.B, axes=axes)
        # fig.colorbar(cl)
        # plt.savefig("B.jpg")
        # fig, axes = plt.subplots()
        # cl = tripcolor(model.H0, axes=axes)
        # fig.colorbar(cl)
        # plt.savefig("H0.jpg")

        if conservation_test:
            with df.CheckpointFile(f"{init_dir}/functions.h5", 'r') as afile:
                H_in = afile.load_function(mesh, "H0", idx=399)
                model.H0.assign(H_in)

        model.beta2.interpolate(df.Constant(100.0))

        z_ela = 0.3

        if conservation_test:
            lapse_rate=0.0
            time_step_factor = 1.01
        else:
            lapse_rate = 2/1000
            time_step_factor = 1.05

        model.adot.dat.data[:] = (((model.B.dat.data[:] + model.H0.dat.data[:])
                                  - z_ela)*lapse_rate)
        print(np.max(model.adot.dat.data[:]))

        S_file = VTKFile(f'{results_dir}/S.pvd')
        B_file = VTKFile(f'{results_dir}/B.pvd')
        Us_file = VTKFile(f'{results_dir}/U_s.pvd')
        H_file = VTKFile(f'{results_dir}/H.pvd')
        N_file = VTKFile(f'{results_dir}/N.pvd')
        adot_file = VTKFile(f'{results_dir}/adot.pvd')

        Q_cg2 = df.VectorFunctionSpace(mesh,"CG",3)
        Q_cg1 = df.FunctionSpace(mesh,"CG",3)
        S_out = df.Function(model.Q_thk,name='S')
        # N_out = df.Function(model.Q_thk,name='N')
        U_s = df.Function(Q_cg1,name='U_s')

        S_out.interpolate(model.S)
        # N_out.interpolate(model.N)
        # U_s.interpolate(model.Ubar0 - 1./4*model.Udef0)
        bla = model.Ubar0 - 1./4*model.Udef0
        ux, uy = bla
        U_s.interpolate(df.sqrt(ux**2+uy**2))

        S_file.write(S_out,time=0.)
        H_file.write(model.H0,time=0.)
        B_file.write(model.B,time=0.)
        Us_file.write(U_s,time=0.)
        adot_file.write(model.adot,time=0.)

        t = 0.0
        t_end = 1
        dt = 0.1
        max_step = 10.0

        # model.calving_factor.assign(10)

        with df.CheckpointFile(f"{results_dir}/functions.h5", 'w') as afile:

            afile.save_mesh(mesh)

            i = 0
            while t<t_end:
                dt = min(dt*time_step_factor,max_step)

                model.adot.dat.data[:] = (((model.B.dat.data[:] + model.H0.dat.data[:])
                                          - z_ela)*lapse_rate)

                converged = model.step(t,
                                       dt,
                                       picard_tol=2e-3,
                                       momentum=0.5,
                                       max_iter=20,
                                       convergence_norm='l2')

                if not converged:
                    dt*=0.5
                    continue
                t += dt
                PETSc.Sys.Print(t,dt,df.assemble(model.H0*df.dx))
                S_out.interpolate(model.S)
                # N_out.interpolate(model.N)
                # U_s.interpolate(model.Ubar0 - 1./4*model.Udef0)
                bla = model.Ubar0 - 1./4*model.Udef0
                ux, uy = bla
                U_s.interpolate(df.sqrt(ux**2+uy**2))

                afile.save_function(model.H0, idx=i)
                afile.save_function(S_out, idx=i)
                # afile.save_function(N_out, idx=i)
                afile.save_function(U_s, idx=i)

                S_file.write(S_out,time=t)
                H_file.write(model.H0,time=t)
                B_file.write(model.B,time=t)
                Us_file.write(U_s,time=t)
                # N_file.write(N_out,time=t)
                adot_file.write(model.adot,time=t)
                i += 1

if __name__=='__main__':
    bc = Glacier('step_10b/results/','step_10/data/')
