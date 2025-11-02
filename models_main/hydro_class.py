import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
import numpy as np
import pandas as pd

class GLADS(object):
    def __init__(self,mesh, results_dir):
        self.mesh     = mesh
        E_phi         = df.FiniteElement("CG", mesh.ufl_cell(), 1)
        E_h           = df.FiniteElement("DG", mesh.ufl_cell(),0)
        E_S           = df.FiniteElement("DGT", mesh.ufl_cell(),0)
        self.elements = [E_phi,E_h,E_S]

        self.V_phi    = df.FunctionSpace(mesh,E_phi)
        self.V_h      = df.FunctionSpace(mesh,E_h)
        self.V_S      = df.FunctionSpace(mesh,E_S)

        # create output files
        self.results_dir = results_dir
        self.outfile_phi = VTKFile(results_dir+'df_phi.pvd')
        self.outfile_h   = VTKFile(results_dir+'df_h.pvd')

    def set_coupler(self,coupler):
        self.coupler = coupler

    def build_variables(self): # shmip_m[shmip_suit]
        # constants
        rho_i = self.coupler.rho_i
        rho_w = self.coupler.rho_w
        g     = self.coupler.g

        # water pressure and effective pressure
        H = self.coupler.H
        B = self.coupler.B
        self.phi = self.coupler.U[0]
        self.P_w = self.phi - rho_w * g * B
        self.N   = rho_i * g * H - self.P_w

    def build_forms(self, m, e_v=1e-4, dt0=3600*2, k_s=0.05, k_c=0.5, h_r=0.5, l_r=5, p_s=0, l_c=10, alpha=1.25, beta=1.5, p=1, q=1, beta2=140,Uhat=1,Nhat=1, transition=False, omega=1/2000):
        s_per_year = 60**2*24*365
        # physical constants
        rho_i = self.coupler.rho_i
        rho_w = self.coupler.rho_w
        g     = self.coupler.g
        n     = self.coupler.n
        A     = df.Constant(self.coupler.A*s_per_year * 2 / n**n)
        L     = self.coupler.L
        ct    = self.coupler.ct
        cw    = self.coupler.cw
        nu    = self.coupler.nu

        # parameters
        k_s   = df.Constant(k_s*s_per_year)       # m^(7/4) kg^(-1/2) -- sheet conductivity
        # k_s = self.k_s = df.Function(self.V_phi).interpolate(k_s*s_per_year)
        k_c   = df.Constant(k_c*s_per_year)       # m^(3/2) kg^(-1/2) -- channel conductivity
        alpha = df.Constant(alpha)       # -                 -- flux exponent
        beta  = df.Constant(beta)        # -                 -- flux exponent
        h_r   = df.Constant(h_r)         # m                 -- bedrock bump height
        l_r   = df.Constant(l_r)         # m                 -- bedrock bump length
        p_s   = df.Constant(p_s)         # -
        l_c   = df.Constant(l_c)         # m                 -- englacial storage ratio
        e_v   = df.Constant(e_v)
        q     = df.Constant(q)
        p     = df.Constant(p)
        beta2 = df.Constant(beta2)
        Uhat  = df.Constant(Uhat)
        Nhat  = df.Constant(Nhat)
        omega = df.Constant(omega)

        # source term
        m = self.m = df.Function(self.V_phi).interpolate(m)

        # initial time step
        dt = self.dt = df.Constant(dt0)

        # variables
        phi = self.phi
        N   = self.N
        h   = self.coupler.U[1]
        S   = self.coupler.U[2]

        # test functions
        xsi = self.coupler.Lambda[0]
        psi = self.coupler.Lambda[1]
        w   = self.coupler.Lambda[2]

        # geometry
        H = self.coupler.H
        B = self.coupler.B

        # initial fields, default
        phi0 = self.phi0 = df.Function(self.V_phi)
        phi0.interpolate(0.5*rho_i * g * H + rho_w * g * B)
        h0   = self.h0   = df.Function(self.V_h)
        h0.vector()[:]   = 0.5*h_r
        S0   = self.S0   = df.Function(self.V_S)
        S0.vector()[:]   = 0.001

        # make first guess equal to initial state (see Burgers tutorial on firedrake documentation)
        self.coupler.U.sub(0).assign(phi0)
        self.coupler.U.sub(1).assign(h0)
        self.coupler.U.sub(2).assign(S0)

        # edge-tangent unit vector
        normal = df.FacetNormal(self.mesh)
        s = df.as_vector([normal[1],-normal[0]])

        # derivative of hydraulic potential along edges
        dphids = df.dot(s,df.grad(phi))

        # derivative of test function along edges
        dxsids = df.dot(s,df.grad(xsi))

        # derivative water pressure along edges
        dPds = df.dot(s,df.grad(self.P_w))

        # edgewise flux
        Q = -k_c*df.max_value(S,1e-15)**alpha*df.max_value(dphids**2, 1e-15)**(beta/2.-1)*dphids
        q_c = -k_s*df.max_value(h,1e-15)**alpha*df.max_value(dphids**2, 1e-15)**(beta/2.-1)*dphids

        # sheet flux
        if transition:
            # Tim Hill's transition model laminar/turbulent
            gradphi = df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(1/2)
            q_s = -nu/(2*omega) * (h_r/h)**(3-2*alpha) * (-1+df.sqrt(1+4*omega/nu*(h/h_r)**(3-2*alpha)*k_s*h**3*gradphi))*df.grad(phi)/gradphi
            self.Re = q_s/nu
        else:
            # 'original' GlaDS
            q_s   = -k_s*df.max_value(h,1e-15)**alpha*df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(beta/2.-1)*df.grad(phi)

        # channel melt rates
        Chi = abs(Q*dphids) + abs(l_c*q_c*dphids)
        # S > Sthresh: f_S = 0 --> for sure f = 1
        # 0 < S < Sthresh: 0 < f_s < 1
        # S =< 0: f_S = 1
        # Sthresh too high: less refreezing, only when channels are bigger
        # Pthresh too high: harder for small channels to re-open
        Sthresh = 5e-2       # below this threshold, allow only reduced freeze-on from sheet input; below 0 don't allow any (given that there is freeze-on)
        Pthresh = 5e-2*3.1536e7       # below this threshold, there is reduced freeze-on; below zero, no freeze-on present (above threshold: normal melt)
        f_S = 1 - df.max_value(0.0, df.min_value(1.0, S/Sthresh))
        f_P = 1 - df.max_value(0.0, df.min_value(1.0, q_c*dPds/Pthresh))
        f   = 1 - f_S*f_P
        Pi = -ct*cw*rho_w*(Q+f*l_c*q_c)*dPds

        # sliding speed from ice flow model
        # u_b = df.sqrt(self.coupler.stokes.u(1)**2 + self.coupler.stokes.v(1)**2 + 1e-10)
        u_b = df.Constant(1e-6*s_per_year)

        # opening and closure for sheets and channels
        O   = df.max_value(u_b*(h_r - h)**(p_s+1)/l_r / h_r**p_s,0)
        C   = A*h*abs(N)**(n-1)*N
        O_c = (Chi-Pi) / (rho_i*L)
        C_c = A*S*abs(N)**(n-1)*N

        R_phi_h = (xsi*e_v/(rho_w*g)*(phi-phi0)/dt  - df.dot(df.grad(xsi),q_s) + xsi * (O-C-m) ) * df.dx
        R_phi_S = df.avg(-dxsids*Q + xsi * O_c*(1-rho_i/rho_w) - xsi * C_c) * df.dS
        R_h     = ((h - h0)/dt - O + C) * psi * df.dx
        R_S     = df.avg(((S-S0)/dt - O_c + C_c)*w) * df.dS + S*w*df.ds # last term is to enforse S = 0 at boundary edges

        self.coupler.R  += R_phi_h + R_phi_S + R_h + R_S

        # boundary conditions
        self.bcs = [df.DirichletBC(self.coupler.V.sub(0), rho_w*g*B, 1)] # id =1 --> part of boundary at the terminus

    def set_timestep(self, dt_):
        self.dt.assign(dt_)

    def update_time_variables(self):
        self.phi0.assign(self.coupler.U.sub(0))
        self.h0.assign(self.coupler.U.sub(1))
        self.S0.assign(self.coupler.U.sub(2))

    def write_variables_pvd(self,t):
        self.outfile_phi.write(df.project(self.coupler.U.sub(0), self.V_phi, name="phi"), time=t)
        self.outfile_h.write(df.project(self.coupler.U.sub(1), self.V_h, name="h"), time=t)

    def save_end_state(self, chk_file, csv_file):
        with CheckpointFile(chk_file, 'w') as afile:
            afile.save_mesh(self.mesh)  # optional
            afile.save_function(self.coupler.U.sub(0), name="phi")
            afile.save_function(self.coupler.U.sub(1), name="h")
        df_S = pd.DataFrame({'S': self.coupler.U.sub(2).vector()[:]})
        df_S.to_csv(csv_file, index=False)

    # for choosing different initial values than the default
    # has to be given as an expression not an array
    def set_initial_phi(self,phi0_):
        self.phi0.interpolate(phi0_)
        self.coupler.U.sub(0).assign(self.phi0)
    def set_initial_h(self,h0_):
        self.h0.interpolate(h0_)
        self.coupler.U.sub(1).assign(self.h0)
    def set_initial_S(self,S0_):
        if type(S0_) == np.ndarray:
            self.S0.vector()[:] = S0_
        else:
            self.S0.interpolate(S0_)
        self.coupler.U.sub(2).assign(self.S0)


class Coupler(object):
    def __init__(self,mesh,hydro):

        # physical constants used in both models
        self.n     = df.Constant(3)           # -              -- flow exponent
        self.A     = df.Constant(3.375e-24)   # Pa^(-n) s^(-1) -- flow constant
        self.g     = df.Constant(9.81)        # m / s^2        -- gravitational acceleration
        self.rho_i = df.Constant(910)         # kg / m^3       -- density of ice

        # only in hydrological model
        self.rho_w = df.Constant(1000)        # kg / m^3       -- density of water
        self.L     = df.Constant(334e3)       # J / kg         -- latent heat of fusion
        self.ct    = df.Constant(7.5e-8)      # K / Pa         -- Clausius-Clapeyron constant
        self.cw    = df.Constant(4220.0)      # J / kg / K     -- specific heat capacity of water
        self.nu    = df.Constant(1.793e-6)

        self.hydro  = hydro

        elements = self.hydro.elements

        E_V      = df.MixedElement(elements)
        self.V   = df.FunctionSpace(mesh,E_V)

        E_cg      = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.Q_cg = df.FunctionSpace(mesh,E_cg)

        self.U      = df.Function(self.V)
        self.Lambda = df.TestFunction(self.V)  # or Function??

        self.R = 0

    def set_geometry(self,B,H):
        self.B = df.Function(self.hydro.V_phi).project(B)
        self.H = df.Function(self.hydro.V_phi).project(H)
        self.S = self.B + self.H



class SHAKTI(object):
    def __init__(self,mesh, results_dir):
        self.mesh  = mesh
        E_h_w      = df.FiniteElement("CG", mesh.ufl_cell(), 1)
        E_h        = df.FiniteElement("CG", mesh.ufl_cell(),1)
        E_K        = df.FiniteElement("CG", mesh.ufl_cell(),1)
        self.E_V   = df.MixedElement([E_h_w,E_h, E_K])
        self.V     = df.FunctionSpace(mesh,self.E_V)
        self.V_h_w = df.FunctionSpace(mesh,E_h_w)
        self.V_h   = df.FunctionSpace(mesh,E_h)
        self.V_K   = df.FunctionSpace(mesh,E_K)

        # create output files
        self.results_dir = results_dir
        self.outfile_h_w = VTKFile(results_dir+results_dir[:-1]+'_h_w.pvd')
        self.outfile_h   = VTKFile(results_dir+results_dir[:-1]+'_h.pvd')
        self.outfile_K   = VTKFile(results_dir+results_dir[:-1]+'_K.pvd')
        self.outfile_N   = VTKFile(results_dir+results_dir[:-1]+'_N.pvd')
        N_out            = df.Function(self.V_h_w)

    def build_variables(self, m_, dt_, thik, bed, e_v_=0.0): # shmip_m[shmip_suit]
        # constants
        rho_i = df.Constant(910)      # kg / m^3
        rho_w = df.Constant(1000)     # kg / m^3
        g     = df.Constant(9.8)      # m / s^2
        L     = df.Constant(334e3)    # J / kg -- latent heat of fusion
        ct    = df.Constant(7.5e-8)   # K / Pa -- Clausius-Clapeyron constant
        cw    = df.Constant(4220.0)   # J / kg / K -- specific heat capacity of water
        h_r   = df.Constant(0.1)      # m
        l_r   = df.Constant(2.0)      # m
        nu    = df.Constant(1.787e-6) # m^2/s -- kinematic viscosity of water
        omega = df.Constant(1e-3)     # - -- controlling nonlinear transition between laminar/turbulent
        G     = df.Constant(0.0)     # W/m^2 -- geothermal heat flux
        A     = df.Constant(5e-25)  # Pa^(-3) s^(-1)
        n     = df.Constant(3)        # -

        e_v   = df.Constant(e_v_)     # -
        m     = self.m = df.Function(self.V_h_w) # m / s
        m.interpolate(m_)

        # initialize bed and ice surface
        B = self.B = df.Function(self.V_h_w)
        self.B.interpolate(bed)
        H = self.H = df.Function(self.V_h_w)
        self.H.interpolate(df.max_value(thik,0))

        # trial and test functions
        U  = self.U = df.Function(self.V)
        h_w, h, K   = df.split(U)
        xsi, psi, w = df.TestFunctions(self.V)

        # initial fields, default
        h_w0 = self.h_w0 = df.Function(self.V_h_w)
        # h_w0.interpolate(1e-4)
        h_w0.interpolate(0.01*H*rho_i/rho_w + B)
        h0   = self.h0   = df.Function(self.V_h)
        h0.interpolate(1e-3)
        K0   = self.K0   = df.Function(self.V_K)
        K0.interpolate(1e-2)

        # make first guess equal to initial state (see Burgers tutorial on firedrake documentation)
        U.sub(0).assign(h_w0)
        U.sub(1).assign(h0)
        U.sub(2).assign(K0)

        # time step
        dt = self.dt = df.Constant(dt_)

        # physical equations #

        # water pressure, hydraulic head and effective pressure
        P_w = rho_w*g*(h_w-B)
        N   = self.N = rho_i*g*H - P_w

        # shear stress and sliding law
        # S     = H + B       # surface elevation
        tau_b = df.Constant(0.0) #  rho_i*g*H*S.dx(0)
        # beta2 = 1e5
        # u_b = tau_b/(N*beta2)
        u_b = df.Constant(1e-6)

        # flux
        q = -K*df.grad(h_w)

        # residual for calculating K
        Re  = K*((df.dot(df.grad(h_w),df.grad(h_w))+1e-10)**0.5)/nu  # Reynolds number
        R_K = w*(K - abs(h)**3*g/(12*nu*(1+omega*Re)))*df.dx

        # opening and closure for sheets and channels
        O   = df.max_value(u_b*(h_r - h)/l_r,0)
        C   = A*h*abs(N)**(n-1)*N
        M   = 1/L * (G + abs(tau_b*u_b) + rho_w*g*df.max_value(df.dot(q,df.grad(h_w)),1e-10) - ct*cw*rho_w*df.max_value(df.dot(q,df.grad(P_w)),-1e-4))

        R_phi_h = (xsi*e_v*(h_w-h_w0)/dt + df.dot(df.grad(xsi),K*df.grad(h_w)) + xsi * (O+M*(1/rho_i-1/rho_w)-C-m) ) * df.dx
        R_h     = ((h - h0)/dt - O - M/rho_i + C) * psi * df.dx
        self.F  = R_phi_h + R_h + R_K

        # boundary conditions
        self.bcs = [df.DirichletBC(self.V.sub(0), B, 1)] # id =1 --> left boundary

    def set_timestep(self, dt_):
        self.dt.assign(dt_)

    def update_time_variables(self):
        self.h_w0.assign(self.U.sub(0))
        self.h0.assign(self.U.sub(1))
        self.K0.assign(self.U.sub(2))

    def write_variables_pvd(self,t):
        self.outfile_h_w.write(df.project(self.U.sub(0), self.V_h_w, name="h_w"))
        self.outfile_h.write(df.project(self.U.sub(1), self.V_h, name="h"))
        self.outfile_K.write(df.project(self.U.sub(2), self.V_h_w, name="K"))
        self.outfile_N.write(df.project(self.N, self.V_h_w, name="N"))

    def save_end_state(self, chk_file):
        with CheckpointFile(chk_file, 'w') as afile:
            afile.save_mesh(self.mesh)  # optional
            afile.save_function(self.U.sub(0), name="h_w")
            afile.save_function(self.U.sub(1), name="h")
            afile.save_function(self.U.sub(2), name="K")

    # for choosing different initial values than the default
    # has to be given as an expression not an array
    def set_initial_h_w(self,h_w0_):
        self.h_w0.interpolate(h_w0_)
        self.U.sub(0).assign(self.h_w0)
    def set_initial_h(self,h0_):
        self.h0.interpolate(h0_)
        self.U.sub(1).assign(self.h0)
    def set_initial_K(self,K0_):
        self.K0.interpolate(K0_)
        self.U.sub(2).assign(self.K0)
