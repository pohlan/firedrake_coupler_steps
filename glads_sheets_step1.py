import firedrake as df
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor, triplot
import numpy as np
import matplotlib.pyplot as plt

s_per_day = 3600 * 24

# mesh
nx, ny = 32, 16
Lx, Ly = 100e3, 20e3
mesh = df.RectangleMesh(nx, ny, Lx, Ly, originX=0.0, originY=0)
x, y = df.SpatialCoordinate(mesh)
E    = df.FunctionSpace(mesh, "CG", 1)
V    = E*E

# bed and ice surface
B = df.Function(E)
B.vector()[:] = 0.0
H = df.Function(E)
H.interpolate(6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1)

# trial and test functions
U        = df.Function(V)
phi, h   = df.split(U)
xsi, psi = df.TestFunctions(V)

# constants
rho_i = df.Constant(910)      # kg / m^3
rho_w = df.Constant(1000)     # kg / m^3
g     = df.Constant(9.8)      # m / s^2
k_s   = df.Constant(5e-3)     # m^(7/4) kg^(-1/2)
alpha = df.Constant(1.25)     # -
beta  = df.Constant(1.5)      # -
u_b   = df.Constant(1e-6)     # m / s
h_r   = df.Constant(0.1)      # m
l_r   = df.Constant(2.0)      # m
A     = df.Constant(2.5e-25)  # Pa^(-3) s^(-1)
n     = df.Constant(3)        # -

e_v   = df.Constant(0.0)      # -
m     = df.Constant(7.93e-11) # m / s

# function definitions
def P_i(H):
    return rho_i * g * H

def P_w(phi, B):
    return phi - rho_w * g * B

def N(phi, H, B):
    return P_i(H) - P_w(phi, B)

def q(phi, h):
    return -k_s*(h**2+1e-15)**(alpha/2.)*(df.dot(df.grad(phi),df.grad(phi))+1e-15)**(beta/2.-1)*df.grad(phi)   # if 1e-15 is replaced by e.g. 1e-3 then it doesn't produce the right result

def O(h):
    return np.max(u_b*(h_r - h)/l_r,0)

def C(h, phi, H, B):
    return A*h*abs(N(phi, H, B))**(n-1)*N(phi, H, B)

# initial fields
phi0 = df.Function(E)
phi0.vector()[:] = rho_i * g * H.vector()[:] * 0.5
h0   = df.Function(E)
h0.vector()[:] = 0.05

fig, axes = plt.subplots()
colors = tripcolor(df.project(phi0,E), axes=axes)
fig.colorbar(colors)
plt.savefig("step1_phi0.jpg")

# numerical
dt = s_per_day*10

# residuals
R_phi = ( - df.dot(df.grad(xsi),q(phi,h)) + xsi * (O(h)-C(h,phi,H,B)-m) ) * df.dx
R_h   = ((h - h0)/dt - O(h) + C(h,phi,H,B)) * psi * df.dx
F     = R_phi + R_h

# boundary conditions
bcs = [df.DirichletBC(V.sub(0), df.Constant(0.0), 1)] # id =1 --> left boundary

par = {'snes_converged_reason': None,
       'snes_monitor': None,
       'snes_linesearch_type': 'bt',
       'ksp_type': 'preonly',
       'pc_type': 'lu',
       "snes_rtol": 1e-7,
       "snes_atol": 1e-7,
       'pc_factor_shift_type': 'inblocks',
       'pc_factor_mat_solver_type': 'mumps'}

outfile_phi = VTKFile('step1_phi.pvd')
outfile_h = VTKFile('step1_h.pvd')

# make first guess equal to initial state (see Burgers tutorial on firedrake documentation)
U.sub(0).assign(phi0)
U.sub(1).assign(h0)

t = 0.0
end = s_per_day*365*20
while (t <= end):
    print(t / (3600*24*365))
    df.solve(F == 0, U, bcs=bcs, solver_parameters=par)
    phi0.assign(U.sub(0))
    h0.assign(U.sub(1))
    t += dt
    outfile_phi.write(df.project(U.sub(0), E, name="phi"))
    outfile_h.write(df.project(U.sub(1), E, name="h"))
