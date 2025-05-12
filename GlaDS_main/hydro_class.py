import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
import numpy as np

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)

class GLADS(object):
    def __init__(self,mesh, results_dir):
        self.mesh  = mesh
        E_phi      = df.FiniteElement("CG", mesh.ufl_cell(), 1)
        E_h        = df.FiniteElement("CG", mesh.ufl_cell(),1)
        E_S        = df.FiniteElement("DGT", mesh.ufl_cell(),0)
        self.E_V   = df.MixedElement([E_phi,E_h,E_S])
        self.V     = df.FunctionSpace(mesh,self.E_V)
        self.V_phi = df.FunctionSpace(mesh,E_phi)
        self.V_h   = df.FunctionSpace(mesh,E_h)
        self.V_S   = df.FunctionSpace(mesh,E_S)

        # create output files
        self.results_dir = results_dir
        self.outfile_phi = VTKFile(results_dir+'step4_phi.pvd')
        self.outfile_h   = VTKFile(results_dir+'step4_h.pvd')
        # self.outfile_S   = VTKFile(results_dir+'step2_S.pvd')
        # self.outfile_Q   = VTKFile(results_dir+'step2_Q.pvd')

    def build_variables(self, m_, dt_): # shmip_m[shmip_suit]
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
        m     = df.Constant(m_) # m / s

        # bed and ice surface
        x, y = df.SpatialCoordinate(self.mesh)
        B = self.B = df.Function(self.V_phi)
        B.vector()[:] = 0.0
        H = self.H = df.Function(self.V_phi)
        H.interpolate(6*( df.sqrt(x+5e3) - df.sqrt(5e3) ) + 1)

        # trial and test functions
        U  = self.U = df.Function(self.V)
        phi, h, S   = df.split(U)
        xsi, psi, w = df.TestFunctions(self.V)

        # initial fields, default
        phi0 = self.phi0 = df.Function(self.V_phi)
        phi0.vector()[:] = rho_i * g * H.vector()[:] * 0.5
        h0   = self.h0   = df.Function(self.V_h)
        h0.vector()[:]   = 0.5*h_r
        S0   = self.S0   = df.Function(self.V_S)
        S0.vector()[:]   = 0.001

        # make first guess equal to initial state (see Burgers tutorial on firedrake documentation)
        U.sub(0).assign(phi0)
        U.sub(1).assign(h0)
        U.sub(2).assign(S0)

        # time step
        dt = self.dt = df.Constant(dt_)

        # physical equations #

        # water pressure and effective pressure
        P_w = phi - rho_w * g * B
        N = self.N = rho_i * g * H - P_w

        # Edge-tangent unit vector
        normal = df.FacetNormal(self.mesh)
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
        R_h     = ((h - h0)/dt - O + C) * psi * df.dx
        R_S     = df.avg(((S-S0)/dt - O_c + C_c)*w) * df.dS + S*w*df.ds # last term is to enforse S = 0 at boundary edges
        self.F  = R_phi_h + R_phi_S + R_h + R_S

        # trick for saving Q
        # p       = df.TestFunction(V_S)
        # dQ      = df.Function(V_S)
        # F_Q     = df.avg((dQ-abs(Q))*p)*df.dS + (dQ-abs(Q))*p*df.ds

        # boundary conditions
        self.bcs = [df.DirichletBC(self.V.sub(0), df.Constant(0.0), 1)] # id =1 --> left boundary

    def set_timestep(self, dt_):
        self.dt.assign(dt_)

    def update_time_variables(self):
        self.phi0.assign(self.U.sub(0))
        self.h0.assign(self.U.sub(1))
        self.S0.assign(self.U.sub(2))

    def write_variables_pvd(self,t):
        self.outfile_phi.write(df.project(self.U.sub(0), self.V_phi, name="phi"))
        self.outfile_h.write(df.project(self.U.sub(1), self.V_h, name="h"))

    def save_end_state(self, chk_file):
        with CheckpointFile(chk_file, 'w') as afile:
            afile.save_mesh(self.mesh)  # optional
            afile.save_function(self.U.sub(0), name="phi")
            afile.save_function(self.U.sub(1), name="h")

    # for choosing different initial values than the default
    # has to be given as an expression not an array
    def set_initial_phi(self,phi0_):
        self.phi0.interpolate(phi0_)
        self.U.sub(0).assign(self.phi0)
    def set_initial_h(self,h0_):
        self.h0.interpolate(h0_)
        self.U.sub(1).assign(self.h0)
    def set_initial_S(self,S0_):
        if type(S0_) == np.ndarray:
            self.S0.vector()[:] = S0_
        else:
            self.S0.interpolate(S0_)
        self.U.sub(2).assign(self.S0)
