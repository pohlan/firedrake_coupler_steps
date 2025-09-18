import firedrake as df
from firedrake.output import VTKFile
from firedrake.checkpointing import CheckpointFile
import numpy as np
import pandas as pd

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
        self.outfile_phi = VTKFile(results_dir+results_dir[:-1]+'_phi.pvd')
        self.outfile_h   = VTKFile(results_dir+results_dir[:-1]+'_h.pvd')
        # self.outfile_S   = VTKFile(results_dir+'step2_S.pvd')
        # self.outfile_Q   = VTKFile(results_dir+'step2_Q.pvd')

    def build_variables(self, m_, dt_, thik, bed, e_v_=0.0): # shmip_m[shmip_suit]
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

        e_v   = df.Constant(e_v_)     # -
        m     = self.m = df.Function(self.V_phi) # m / s
        m.interpolate(m_)

        # initialize bed and ice surface
        B = self.B = df.Function(self.V_phi)
        self.B.interpolate(bed)
        H = self.H = df.Function(self.V_phi)
        self.H.interpolate(df.max_value(thik,0))

        # trial and test functions
        U  = self.U = df.Function(self.V)
        phi, h, S   = df.split(U)
        xsi, psi, w = df.TestFunctions(self.V)

        # initial fields, default
        phi0 = self.phi0 = df.Function(self.V_phi)
        phi0.interpolate(rho_i * g * H + rho_w * g * B)
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
        Q = -k_c*df.max_value(S,1e-15)**alpha*df.max_value(dphids**2, 1e-15)**(beta/2.-1)*dphids
        q_c = -k_s*df.max_value(h,1e-15)**alpha*df.max_value(dphids**2, 1e-15)**(beta/2.-1)*dphids

        # Sheet flux
        q   = -k_s*df.max_value(h,1e-15)**alpha*df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(beta/2.-1)*df.grad(phi)

        # Channel melt rates
        Chi = abs(Q*dphids) + abs(l_c*q_c*dphids)
        # S > Sthresh: f_S = 0 --> for sure f = 1
        # 0 < S < Sthresh: 0 < f_s < 1
        # S =< 0: f_S = 1
        # Sthresh too high: less refreezing, only when channels are bigger
        # Pthresh too high: harder for small channels to re-open
        Sthresh = 5e-2       # below this threshold, allow only reduced freeze-on from sheet input; below 0 don't allow any (given that there is freeze-on)
        Pthresh = 5e-2       # below this threshold, there is reduced freeze-on; below zero, no freeze-on present (above threshold: normal melt)
        f_S = 1 - df.max_value(0.0, df.min_value(1.0, S/Sthresh))
        f_P = 1 - df.max_value(0.0, df.min_value(1.0, q_c*dPds/Pthresh))
        f   = 1 - f_S*f_P
        Pi = -ct*cw*rho_w*(Q+f*l_c*q_c)*dPds
        # Pi = 0 is no problem; f=1-f_P works as well; f=1-f_S doesn't if the threshold is too low

        # opening and closure for sheets and channels
        O   = df.max_value(u_b*(h_r - h)/l_r,0)
        C   = A*h*abs(N)**(n-1)*N
        O_c = (Chi-Pi) / (rho_i*L)
        C_c = A*S*abs(N)**(n-1)*N

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
        self.bcs = [df.DirichletBC(self.V.sub(0), rho_w*g*B, 1)] # id =1 --> left boundary

    def set_timestep(self, dt_):
        self.dt.assign(dt_)

    def update_time_variables(self):
        self.phi0.assign(self.U.sub(0))
        self.h0.assign(self.U.sub(1))
        self.S0.assign(self.U.sub(2))

    def write_variables_pvd(self,t):
        self.outfile_phi.write(df.project(self.U.sub(0), self.V_phi, name="phi"))
        self.outfile_h.write(df.project(self.U.sub(1), self.V_h, name="h"))

    def save_end_state(self, chk_file, csv_file):
        with CheckpointFile(chk_file, 'w') as afile:
            afile.save_mesh(self.mesh)  # optional
            afile.save_function(self.U.sub(0), name="phi")
            afile.save_function(self.U.sub(1), name="h")
        df_S = pd.DataFrame({'S': self.U.sub(2).vector()[:]})
        df_S.to_csv(csv_file, index=False)

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
