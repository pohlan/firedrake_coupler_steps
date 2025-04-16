import firedrake as df
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor, triplot
import numpy as np
import matplotlib.pyplot as plt
# from firedrake.checkpointing import CheckpointFile

s_per_day = 3600 * 24
results_dir = 'step_2/'

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)

# mesh
nx, ny = 32, 16
Lx, Ly = 100e3, 20e3
mesh   = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
x, y   = df.SpatialCoordinate(mesh)
E_phi  = df.FiniteElement("CG", mesh.ufl_cell(), 1)
E_h    = df.FiniteElement("DG", mesh.ufl_cell(),0)
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
m     = df.Constant(5.79e-8) # m / s

# initial fields
phi0 = df.Function(V_phi)
phi0.vector()[:] = rho_i * g * H.vector()[:] * 0.5
h0   = df.Function(V_h)
h0.vector()[:] = 0.05
S0   = df.Function(V_S)
S0.vector()[:] = 0.001

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
R_S     = df.avg(((S-S0)/dt - O_c + C_c)*w) * df.dS + S*w*df.ds #?? What is the last term for ??
F       = R_phi + R_h + R_S

# boundary conditions
bcs = [df.DirichletBC(V.sub(0), df.Constant(0.0), 1)] # id =1 --> left boundary

par = {"snes_type": "vinewtonrsls",
       "pc_factor_mat_solver_type": "mumps", # ?
       "snes_rtol": 1e-5,
       "snes_atol": 1e-5,
       "snes_max_it": 40,
       "report": True,
       "snes_monitor": None,
       "error_on_nonconvergence": True}

outfile_phi = VTKFile(results_dir+'step2_phi.pvd')
outfile_h   = VTKFile(results_dir+'step2_h.pvd')
outfile_S   = VTKFile(results_dir+'step2_S.pvd')

# make first guess equal to initial state (see Burgers tutorial on firedrake documentation)
U.sub(0).assign(phi0)
U.sub(1).assign(h0)
U.sub(2).assign(S0)

t = 0.0
end = s_per_day*365*5
dt_max = s_per_day*5
dt_min = s_per_day*1e-5
timestep_increase_fraction = 1.01
timestep_reduction_fraction = 0.5
while (t <= end):
    try:
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
    # with CheckpointFile("output.h5", 'w') as chk:
    #     chk.set_timestep(0)
    #     chk.write_function(U.sub(2))
    #     chk.set_timestep(1)
    #     chk.write_function(f)
    # outfile_S.write(df.project(U.sub(2), V_CR, name="S"))
    except RuntimeError:
        dt = dt*timestep_reduction_fraction
        if dt < dt_min: break
        print('Convergence not achieved.  Reducting time step to {0} and trying again'.format(dt))
