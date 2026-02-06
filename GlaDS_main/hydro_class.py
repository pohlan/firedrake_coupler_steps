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
        self.outfile_q = VTKFile(results_dir+results_dir[:-1]+'_q.pvd')
        self.outfile_h   = VTKFile(results_dir+results_dir[:-1]+'_h.pvd')
        self.outfile_N   = VTKFile(results_dir+results_dir[:-1]+'_N.pvd')

    def build_variables(self, m_, df_moul, dt_, surface, bed, e_v_=0.0): # shmip_m[shmip_suit]
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
        Am    = df.Constant(10.0)

        e_v   = df.Constant(e_v_)     # -
        m     = self.m = df.Function(self.V_phi) # m / s
        m.interpolate(m_)

        # initialize bed and ice surface
        x, y = df.SpatialCoordinate(self.mesh)
        B = self.B = df.Function(self.V_phi)
        self.B.interpolate(bed(x,y))
        H = self.H = df.Function(self.V_phi)
        self.H.interpolate(df.max_value(surface(x,y)-bed(x,y),0))

        # trial and test functions
        U  = self.U = df.Function(self.V)
        phi, h, S   = df.split(U)
        xsi, psi, w = df.TestFunctions(self.V)

        # make melt input to only specific nodes where there is a moulin
        # source_location = [[xx,yy] for (xx,yy) in zip(df_moul.x,df_moul.y)]
        # v_mesh = df.VertexOnlyMesh(self.mesh, source_location)
        # V_s = df.FunctionSpace(v_mesh, "DG", 0)
        # delta = df.Function(V_s).interpolate(df_moul.m[0]*s_per_year)
        # v_test = df.TestFunction(V_s)
        # source_cofunction = df.assemble(delta * v_test * df.dx)
        # Qs = df.Cofunction(self.V.dual())
        # Qs.sub(0).interpolate(source_cofunction)
        # # f.interpolate(conditional('distance from (x,y)'))
        # # Qs = df.project(source_cofunction, self.V.subfunctions[0])
        # v_check = df.Function(self.V.sub(1))
        # v_check.assign(Qs.sub(1).riesz_representation())
        # VTKFile("point_source.pvd").write(v_check)

        # 'width' of delta, regularization
        sources = [([xx,yy], mm) for (xx,yy,mm) in zip(df_moul.x,df_moul.y,df_moul.m)]
        eps = df.Constant(1.5e3)  # should be slightly larger than grid resolution
        # sum the kernels
        x = df.SpatialCoordinate(self.mesh)
        # Qm = 0
        delta_moul = 0
        Q_in = df.Constant(df_moul.m[0])  # assumes the same input for each moulin
        for x0_val, Q_val in sources:
            x0 = df.Constant(x0_val)
            # Q_in  = df.Constant(Q_val)

            r2 = df.dot(x - x0, x - x0)
            # Normalized 2D kernel
            delta_eps = (1 / (df.pi * eps**2)) * df.exp(-r2 / eps**2)

            # Qm += Q_in * delta_eps
            delta_moul += delta_eps

        Qm_save = df.Function(self.V_phi)
        Qm_save.interpolate(delta_moul)
        VTKFile(self.results_dir+"point_source.pvd").write(Qm_save)

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
        self.Q = Q = -k_c*df.max_value(S,1e-15)**alpha*df.max_value(dphids**2, 1e-15)**(beta/2.-1)*dphids
        q_c = -k_s*df.max_value(h,1e-15)**alpha*df.max_value(dphids**2, 1e-15)**(beta/2.-1)*dphids

        # Sheet flux
        q = self.q = -k_s*df.max_value(h,1e-15)**alpha*df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(beta/2.-1)*df.grad(phi)

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
        C   = A*h*abs(N)**(n-1)*df.max_value(N,10000)
        O_c = (Chi-Pi) / (rho_i*L)
        C_c = A*S*abs(N)**(n-1)*df.max_value(N,10000)

        R_phi_h = (xsi*e_v/(rho_w*g)*(phi-phi0)/dt  - df.dot(df.grad(xsi),q) + xsi * (O-C-m) ) * df.dx(degree=3)
        R_phi_S = df.avg(-dxsids*Q + xsi * O_c*(1-rho_i/rho_w) - xsi * C_c) * df.dS
        R_phi_M = - xsi*(-Am/(rho_w*g)*(phi-phi0)/dt + Q_in)*delta_moul*df.dx
        R_h     = ((h - h0)/dt - O + C) * psi * df.dx
        R_S     = df.avg(((S-S0)/dt - O_c + C_c)*w) * df.dS + S*w*df.ds # last term is to enforse S = 0 at boundary edges
        self.F  = R_phi_h + R_phi_S + R_h + R_S + R_phi_M

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
        # qq = (self.q[0]**2+self.q[1]**2)**0.5
        V_vec = df.VectorFunctionSpace(self.mesh, "CG", 1)
        self.outfile_q.write(df.project(self.q, V_vec, name="q_s"))
        self.outfile_h.write(df.project(self.U.sub(1), self.V_h, name="h"))
        self.outfile_N.write(df.project(self.N, self.V_phi, name="N"))

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
