import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
from firedrake.pyplot import tripcolor, triplot
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.interpolate as itp

s_per_day = 3600 * 24
results_dir = 'step_2/'
chk_file    = results_dir + "initial_fields_A1.h5"

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)

# TODO : include S in the comparison !!

shmip_suit = "A1"
shmip_m = {"A1" : 7.93e-11,
           "A2" : 1.59e-9,
           "A3" : 5.79e-9,
           "A4" : 2.5e-8,
           "A5" : 4.5e-8,
           "A6" : 5.79e-7}

# mesh
nx, ny = 75, 25
Lx, Ly = 100e3, 20e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
x, y   = df.SpatialCoordinate(mesh)
E_phi  = df.FiniteElement("CG", mesh.ufl_cell(), 1)
E_h    = df.FiniteElement("CG", mesh.ufl_cell(),1)
E_S    = df.FiniteElement("DGT", mesh.ufl_cell(),0)
E_V    = df.MixedElement([E_phi,E_h,E_S])
V      = df.FunctionSpace(mesh,E_V)
V_phi  = df.FunctionSpace(mesh,E_phi)
V_h    = df.FunctionSpace(mesh,E_h)
V_S    = df.FunctionSpace(mesh,E_S)

# bed and ice surface
B = df.Function(V_phi)
B.vector()[:] = 0.0
H = df.Function(V_phi)
H.interpolate(6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1)

# trial and test functions
U           = df.Function(V)
phi, h, S   = df.split(U)
xsi, psi, w = df.TestFunctions(V)

# constants
rho_i = df.Constant(910)      # kg / m^3
rho_w = df.Constant(1000)     # kg / m^3
g     = df.Constant(9.8)      # m / s^2
L     = df.Constant(334e3)    # J / kg -- latent heat of fusion
ct    = df.Constant(7.5e-8)   # K / Pa -- Clausius-Clapeyron constant
cw    = df.Constant(4220.0)   # J / kg / K -- specific heat capacity of water
k_s   = df.Constant(5e-3)     # m^(7/4) kg^(-1/2) -- sheet conductivity
k_c   = df.Constant(0.1)      # m^(3/2) kg^(-1/2)
alpha = df.Constant(1.25)     # -
beta  = df.Constant(1.5)      # -
u_b   = df.Constant(1e-6)     # m / s
h_r   = df.Constant(0.1)      # m
l_r   = df.Constant(2.0)      # m
l_c   = df.Constant(2.0)      # m
A     = df.Constant(2.5e-25)  # Pa^(-3) s^(-1)
n     = df.Constant(3)        # -

e_v   = df.Constant(0.0)      # -
m     = df.Constant(shmip_m[shmip_suit]) # m / s

# initial fields
phi0 = df.Function(V_phi)
h0   = df.Function(V_h)

if shmip_suit == "A1": # for A1 scenario (low melt), initialize with some arbitrary numbers
    phi0.vector()[:] = rho_i * g * H.vector()[:] * 0.5
    h0.vector()[:] = 0.05
else: # otherwise, use steady-state result from A1 test case
    with CheckpointFile(chk_file, 'r') as afile:
        mesh_ = afile.load_mesh()
        phi0_ = afile.load_function(mesh_, "phi")
        h0_   = afile.load_function(mesh_, "h")
        phi0.interpolate(phi0_)
        h0.interpolate(h0_)
        # S0   = afile.load_function(mesh, "S") # doesn't work for DGT
# because trace elements behave weirdly, S is always initialized with an arbitrary number
S0   = df.Function(V_S)
S0.vector()[:] = 0.001 # * np.random.rand(len(meshx_S))

# numerical
dt = s_per_day*0.5

# physical equations #

# water pressure and effective pressure
P_w = phi - rho_w * g * B
N   = rho_i * g * H - P_w

# Edge-tangent unit vector
normal = df.FacetNormal(mesh)
s = df.as_vector([normal[1],-normal[0]])

# derivative of hydraulic potential along edges
dphids = df.dot(s,df.grad(phi))

# derivative of test function along edges
dxsids = df.dot(s,df.grad(xsi))

# derivative water pressure along edges
dPds = df.dot(s,df.grad(P_w))

# Edgewise flux
Q = -k_c*Max(S,1e-15)**alpha*Max(dphids**2, 1e-15)**(beta/2.-1)*dphids
q_c = -k_s*Max(h,1e-15)**alpha*Max(dphids**2, 1e-15)**(beta/2.-1)*dphids

# Sheet flux
q   = -k_s*(h+1e-15)**alpha*(df.dot(df.grad(phi),df.grad(phi))+1e-15)**(beta/2.-1)*df.grad(phi)

# Channel melt rates
Chi = abs(Q*dphids) + abs(l_c*q_c*dphids)
f   = Max(Max(df.sign(S),0), Max(df.sign(q_c*dPds),0))
Pi = -ct*cw*rho_w*(Q+f*l_c*q_c)*dPds

# opening and closure for sheets and channels
O   = Max(u_b*(h_r - h)/l_r,0)
C   = A*h*abs(N)**(n-1)*Max(N,1000)
O_c = (Chi-Pi) / (rho_i*L)
C_c = A*S*abs(N)**(n-1)*Max(N,1000)

R_phi_h = (xsi*e_v/(rho_w*g)*(phi-phi0)/dt  - df.dot(df.grad(xsi),q) + xsi * (O-C-m) ) * df.dx
R_phi_S = df.avg(-dxsids*Q + xsi * O_c*(1-rho_i/rho_w) - xsi * C_c) * df.dS
R_phi   = R_phi_h + R_phi_S
R_h     = ((h - h0)/dt - O + C) * psi * df.dx
R_S     = df.avg(((S-S0)/dt - O_c + C_c)*w) * df.dS + S*w*df.ds # last term is to enforse S = 0 at boundary edges
F       = R_phi + R_h + R_S

# p       = df.TestFunction(V_S)
# dQ      = df.Function(V_S)
# F_Q     = df.avg((dQ-abs(Q))*p)*df.dS + (dQ-abs(Q))*p*df.ds

# boundary conditions
bcs = [df.DirichletBC(V.sub(0), df.Constant(0.0), 1)] # id =1 --> left boundary

par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps", # ?
       "snes_rtol": 1e-5,
       "snes_atol": 1e-5,
       "snes_max_it": 1000,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

outfile_phi = VTKFile(results_dir+'step2_phi.pvd')
outfile_h   = VTKFile(results_dir+'step2_h.pvd')
outfile_S   = VTKFile(results_dir+'step2_S.pvd')
outfile_Q   = VTKFile(results_dir+'step2_Q.pvd')

# make first guess equal to initial state (see Burgers tutorial on firedrake documentation)
U.sub(0).assign(phi0)
U.sub(1).assign(h0)
U.sub(2).assign(S0)

t = 0.0
end = s_per_day*365*20
dt_max = s_per_day*5
dt_min = s_per_day*1e-5
timestep_increase_fraction = 1.1
timestep_reduction_fraction = 0.5
while (t <= end):
    print([t / (3600*24*365), dt / s_per_day])
    df.solve(F == 0, U, bcs=bcs, solver_parameters=par)
    phi0.assign(U.sub(0))
    h0.assign(U.sub(1))
    S0.assign(U.sub(2))
    t += dt
    dt = min(dt*timestep_increase_fraction,dt_max)
    outfile_phi.write(df.project(U.sub(0), V_phi, name="phi"))
    outfile_h.write(df.project(U.sub(1), V_h, name="h"))
    # outfile_S.write(df.project(S, V_CG, name="S"))

    # Doug's trick to save Q (no easier way?)
    # df.solve(F_Q == 0, dQ)
    # outfile_Q.write(df.project(dQ, V_S, name="Q"))

print(np.max(U.sub(0).vector()[:]))

# save end result so it can serve as initial field for next simulation
chk_file_save = results_dir + "initial_fields_"+shmip_suit+".h5"
with CheckpointFile(chk_file, 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(U.sub(0), name="phi")
    afile.save_function(U.sub(1), name="h")

# test against GlaDS-matlab SHMIP results
fl = "SHMIP_results/mw/"+shmip_suit+"_mw.nc"
ds_mw = xr.open_dataset(fl)
x_mw = ds_mw.coords1.data[0]
y_mw = ds_mw.coords1.data[1]
V_mesh = df.VectorFunctionSpace(mesh, E_phi)
X = df.assemble(interpolate(mesh.coordinates, V_mesh))
meshx = X.dat.data[:,0]
meshy = X.dat.data[:,1]

F_mw = df.Function(V_phi)
F_diff = df.Function(V_phi)
for (f_mw, F_sol) in zip([ds_mw.N, ds_mw.h], [N, h]):
    F_mw.vector()[:] = itp.griddata((x_mw,y_mw), f_mw.data[0], (meshx,meshy))
    F_diff.interpolate(F_mw-F_sol)
    outfile_t = VTKFile(shmip_suit+"_glads-matlab.pvd")
    outfile_t.write(df.project(rho_i * g * H-F_mw, V_phi, name="phi"))
    print(np.max(np.abs(F_diff.vector()[:]/F_mw.vector()[:])))
    # assert np.max(np.abs(F_diff.vector()[:]/F_mw.vector()[:])) < 0.05

# save plot of steady state solution including S
Vmesh_S = df.VectorFunctionSpace(mesh, E_S)
X = df.assemble(interpolate(mesh.coordinates,Vmesh_S))
meshx_S = X.dat.data[:,0]
meshy_S = X.dat.data[:,1]

plt.figure()
plt.scatter(meshx_S, meshy_S, 10, U.sub(2).vector()[:])
plt.colorbar()
plt.savefig("S_"+shmip_suit+".jpg")

plt.figure()
plt.scatter(meshx, meshy, 10, U.sub(0).vector()[:])
plt.colorbar()
plt.savefig("phi_"+shmip_suit+".jpg")

plt.figure()
plt.scatter(meshx, meshy, 20, U.sub(1).vector()[:])
plt.colorbar()
plt.savefig("h_"+shmip_suit+".jpg")
