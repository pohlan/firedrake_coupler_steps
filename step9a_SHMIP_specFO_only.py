import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, triplot
from models_main.SpecFO import SpecFO, Coupler

results_dir = 'step_9a/results/'
mesh = df.Mesh("valley.msh")
x, y = df.SpatialCoordinate(mesh)
stokes = SpecFO(mesh)
coupler = Coupler(mesh, stokes)

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

B = df.Function(coupler.Q_cg).interpolate(bed(x,y))
H = df.Function(coupler.Q_cg).interpolate(surface(x,y)-bed(x,y))

thklim = 0
thklim = 10
Htemp = H.vector().get_local()
Htemp[Htemp<thklim] = thklim
# Htemp[np.isnan(Htemp)] = thklim
H.vector().set_local(Htemp)

################
fig, axes = plt.subplots()
cl = tripcolor(B, axes=axes)
fig.colorbar(cl)
plt.savefig(f"{results_dir}B.jpg")
# thickness
fig, axes = plt.subplots()
cl = tripcolor(H, axes=axes)
fig.colorbar(cl)
plt.savefig(f"{results_dir}H.jpg")
################

coupler.set_geometry(B,H)

stokes.set_coupler(coupler)
stokes.build_variables(p=0.6,q=0.6,beta2=140,results_dir=results_dir)
stokes.build_forms()

solver_params = {"snes_type": "vinewtonrsls",#newton
                 "pc_factor_mat_solver_type": "mumps", # ?
                 "snes_rtol": 1e-2,
                 "snes_atol": 1e-2,
                 "snes_max_it": 100,
                 "report": True,
                 "snes_monitor": None,
                 "error_on_nonconvergence": True}

df.solve(coupler.R == 0, coupler.U, solver_parameters=solver_params)
stokes.write_variables(0)
