import firedrake as df
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor, triplot
import numpy as np
import matplotlib.pyplot as plt

s_per_day = 3600 * 24
results_dir = 'step_2/'

# mesh
nx, ny = 64, 32
Lx, Ly = 100e3, 20e3
mesh = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
x, y = df.SpatialCoordinate(mesh)
# E    = df.FunctionSpace(mesh, "CG", 1)
# E_CR = df.FunctionSpace(mesh, "CR", 1)  # for channels, defined on edges
E_CG = df.FiniteElement("CG", mesh.ufl_cell(), 1)
E_CR = df.FiniteElement("CR", mesh.ufl_cell(),1)
E_V  = df.MixedElement([E_CG,E_CG,E_CR])
V    = df.FunctionSpace(mesh,E_V)
V_CG = df.FunctionSpace(mesh,E_CG)
V_CR = df.FunctionSpace(mesh,E_CR)

# bed and ice surface
B = df.Function(V_CG)
B.vector()[:] = 0.0
H = df.Function(V_CG)
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
A     = df.Constant(2.5e-25)  # Pa^(-3) s^(-1)
n     = df.Constant(3)        # -

e_v   = df.Constant(0.0)      # -
m     = df.Constant(7.93e-11) # m / s

# pressure
def P_i(H):
    return rho_i * g * H

def P_w(phi, B):
    return phi - rho_w * g * B

def N(phi, H, B):
    return P_i(H) - P_w(phi, B)

# sheets
def q(phi, h):
    return -k_s*(h**2+1e-15)**(alpha/2.)*(df.dot(df.grad(phi),df.grad(phi))+1e-15)**(beta/2.-1)*df.grad(phi)   # if 1e-15 is replaced by e.g. 1e-3 then it doesn't produce the right result

def O(h):
    return np.max(u_b*(h_r - h)/l_r,0)

def C(h, phi, H, B):
    return A*h*abs(N(phi, H, B))**(n-1)*N(phi, H, B)

# channels
def dXds(X):
    normal = df.FacetNormal(mesh)
    s = df.as_vector([normal[1],-normal[0]])
    return df.dot(s,df.grad(X))

def Q(dphi_ds, S):
    return -k_c*(S**2+1e-15)**(alpha/2.)*(dphi_ds**2 + 1e-15)**(beta/2. - 1)*dphi_ds

def Chi(Q_, dphi_ds):
    return abs(Q_*dphi_ds) # what about contribution from sheets?

def Pi(Q_, dphi_ds, B):
    dB_ds = dXds(B)
    return -ct*cw*rho_w * Q_ * (dphi_ds - rho_w*g*dB_ds)  # contribution from sheets?

def O_c(Q_, dphi_ds, B):
    return (Chi(Q_, dphi_ds) -Pi(Q_, dphi_ds, B)) / (rho_i*L)

def C_c(S, phi, H, B):
    return A*S*abs(N(phi, H, B))**(n-1)*N(phi, H, B)

def R_phi_c(phi, S, xsi, B, H):
    dphi_ds = dXds(phi)
    dxsi_ds = dXds(xsi)
    Q_      = Q(dphi_ds, S)
    return dxsi_ds*Q_ + xsi * O_c(Q_, dphi_ds, B)*(1-rho_i/rho_w) - xsi * C_c(S, phi, H, B)

# initial fields
phi0 = df.Function(V_CG)
phi0.vector()[:] = rho_i * g * H.vector()[:] * 0.5
h0   = df.Function(V_CG)
h0.vector()[:] = 0.05
S0   = df.Function(V_CR)
S0.vector()[:] = 0.001

# numerical
dt = s_per_day*10

# residuals
dphi_ds = dXds(phi)
Q_      = Q(dphi_ds, S)
# O_c_    = (Chi(Q_, dphi_ds) -Pi(Q_, dphi_ds, B)) / (rho_i*L)
R_phi   = ( - df.dot(df.grad(xsi),q(phi,h)) + xsi * (O(h)-C(h,phi,H,B)-m) ) * df.dx + R_phi_c(phi, S, xsi, B, H)('+') * df.dS
R_h     = ((h - h0)/dt - O(h) + C(h,phi,H,B))* psi * df.dx
R_S     = (((S-S0)/dt - O_c(Q_, dphi_ds, B) + C_c(S, phi, H, B))*w)('+') * df.dS  + S*w*df.ds #?? What is the last term for ??
F       = R_phi + R_h + R_S

# boundary conditions
bcs = [df.DirichletBC(V.sub(0), df.Constant(0.0), 1)] # id =1 --> left boundary

# par = {'snes_converged_reason': None,
#        'snes_monitor': None,
#        'snes_linesearch_type': 'bt',
#        'ksp_type': 'preonly',
#        'pc_type': 'lu',
#        "snes_rtol": 1e-7,
#        "snes_atol": 1e-7,
#        'pc_factor_shift_type': 'inblocks',
#        'pc_factor_mat_solver_type': 'mumps'}
par = {"snes_type": "vinewtonrsls",#newton
       "pc_factor_mat_solver_type": "mumps", # ?
       "snes_rtol": 1e-5,
       "snes_atol": 1e-5,
       "snes_max_it": 12,
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
end = s_per_day*365*2
while (t <= end):
    print(t / (3600*24*365))
    df.solve(F == 0, U, bcs=bcs, solver_parameters=par)
    phi0.assign(U.sub(0)) # technically only necessary when there's a time derivative term with e_v
    h0.assign(U.sub(1))
    S0.assign(U.sub(2))
    t += dt
    outfile_phi.write(df.project(U.sub(0), V_CG, name="phi"))
    outfile_h.write(df.project(U.sub(1), V_CG, name="h"))
    outfile_S.write(df.project(U.sub(2), V_CR, name="S"))
