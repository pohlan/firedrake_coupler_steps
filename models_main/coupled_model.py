import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
from firedrake.checkpointing import CheckpointFile
import numpy as np
import pandas as pd
import models_main.helpers as hlp

class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])


# PERFORMS GAUSSIAN QUADRATURE FOR ARBITRARY FUNCTION OF SIGMA, QUAD POINTS, AND WEIGHTS
class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

from numpy.polynomial.legendre import leggauss
def half_quad(order):
    points,weights = leggauss(order)
    points=points[(order-1)//2:]
    weights=weights[(order-1)//2:]
    weights[0] = weights[0]/2
    return points,weights

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)
def Min(a, b): return (a+b-abs(a-b))/df.Constant(2)
def Softplus(a,b,alpha=1): return Max(a,b) + 1./alpha*df.ln(1 + df.exp(-abs(a-b)*alpha))


class SpecFO(object):
    def __init__(self,mesh,results_dir='./results/'):
        E_cg = self.E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.elements = [E_cg]*4

        # create output files
        self.results_dir = results_dir
        self.Us_file = df.VTKFile(results_dir+'df_U_s.pvd')
        self.Ub_file = df.VTKFile(results_dir+'df_U_b.pvd')

    def set_coupler(self,coupler):
        self.coupler = coupler

    def build_variables(self):
        n = self.coupler.n

        self.ubar = self.coupler.U[3]
        self.vbar = self.coupler.U[4]
        self.udef = self.coupler.U[5]
        self.vdef = self.coupler.U[6]

        self.lamdabar_x = self.coupler.Lambda[3]
        self.lamdabar_y = self.coupler.Lambda[4]
        self.lamdadef_x = self.coupler.Lambda[5]
        self.lamdadef_y = self.coupler.Lambda[6]

        # TEST FUNCTION COEFFICIENTS
        coef  = [lambda s:1.0, lambda s:1./(n+1)*((n+2)*s**(n+1) - 1)]
        dcoef = [lambda s:0,   lambda s:(n+2)*s**n]

        u_       = [self.ubar,       self.udef]
        v_       = [self.vbar,       self.vdef]
        lamda_x_ = [self.lamdabar_x, self.lamdadef_x]
        lamda_y_ = [self.lamdabar_y, self.lamdadef_y]

        self.u       = VerticalBasis(u_,coef,dcoef)
        self.v       = VerticalBasis(v_,coef,dcoef)
        self.lamda_x = VerticalBasis(lamda_x_,coef,dcoef)
        self.lamda_y = VerticalBasis(lamda_y_,coef,dcoef)

        self.U_b = df.as_vector([self.u(1),self.v(1)])
        # self.tau_b = df.Function(self.coupler.Q_cg)

        # self.Us = df.project(df.as_vector([self.u(0),self.v(0)]), self.Q)
        self.Us = df.project(df.sqrt(self.u(0)**2+self.v(0)**2), self.coupler.Q_cg)
        # self.Ub = df.project(df.as_vector([self.u(1),self.v(1)]), self.Q)
        self.Ub = df.project(df.sqrt(self.u(1)**2+self.v(1)**2), self.coupler.Q_cg)

    def build_forms(self, p=0.5, q=0.5, eps_reg=1e-5, beta2=140, Uhat=1, Nhat=1):
        n          = self.coupler.n
        g          = self.coupler.g
        rho_i      = self.coupler.rho_i
        # scale ice flow constant to get velocities in m/a not m/s
        s_per_year = 3600*24*365
        Be         = (self.coupler.A*s_per_year)**(-1/n) # Pa^(-n) a^(-1)
        Uhat       = df.Constant(Uhat)
        Nhat       = df.Constant(Nhat)
        # ice flow specific constants
        eps_reg    = df.Constant(eps_reg)
        p          = df.Constant(p)
        q          = df.Constant(q)
        beta2      = df.Constant(beta2)

        u = self.u
        v = self.v
        lamda_x = self.lamda_x
        lamda_y = self.lamda_y
        H = self.coupler.H
        B = self.coupler.B
        S = self.coupler.S
        N = self.coupler.hydro.N
        # N = 0.5*rho_i*g*H

        def dsdx(s):
            return 1./H*(S.dx(0) - s*H.dx(0))

        def dsdy(s):
            return 1./H*(S.dx(1) - s*H.dx(1))

        def dsdz(s):
            return -1./H

        # 2nd INVARIANT STRAIN RATE
        def epsilon_dot(s):
            return ((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                        +(v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
                        +(u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
                        +0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
                        + ((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
                        + eps_reg)

        # VISCOSITY
        def eta_v(s):
            return Be/2.*epsilon_dot(s)**((1.-n)/(2*n))

        # MEMBRANE STRESSES
        def membrane_xx(s):
            return (lamda_x.dx(s,0) + lamda_x.ds(s)*dsdx(s))*H*(eta_v(s))*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))

        def membrane_xy(s):
            return (lamda_x.dx(s,1) + lamda_x.ds(s)*dsdy(s))*H*(eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

        def membrane_yx(s):
            return (lamda_y.dx(s,0) + lamda_y.ds(s)*dsdx(s))*H*(eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

        def membrane_yy(s):
            return (lamda_y.dx(s,1) + lamda_y.ds(s)*dsdy(s))*H*(eta_v(s))*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))

        # SHEAR STRESSES
        def shear_xz(s):
            return dsdz(s)**2*lamda_x.ds(s)*H*eta_v(s)*u.ds(s)

        def shear_yz(s):
            return dsdz(s)**2*lamda_y.ds(s)*H*eta_v(s)*v.ds(s)

        # DRIVING STRESSES
        def tau_dx(s):
            return rho_i*g*H*S.dx(0)*lamda_x(s)

        def tau_dy(s):
            return rho_i*g*H*S.dx(1)*lamda_y(s)

        # GET QUADRATURE POINTS (THIS SHOULD BE ODD: WILL GENERATE THE GAUSS-LEGENDRE RULE
        # POINTS AND WEIGHTS OF O(n), BUT ONLY THE POINTS IN [0,1] ARE KEPT< DUE TO SYMMETRY.
        points,weights = half_quad(9)

        # INSTANTIATE VERTICAL INTEGRATOR
        vi = VerticalIntegrator(points,weights)

        tau_bx = -beta2*(Max(N,1e4)/Nhat)**p*abs((u(1)**2 + v(1)**2)/Uhat**2 + 1e-2)**((q-1)/2.)*u(1)/Uhat  # does not converge without the Max(N,...)
        tau_by = -beta2*(Max(N,1e4)/Nhat)**p*abs((u(1)**2 + v(1)**2)/Uhat**2 + 1e-2)**((q-1)/2.)*v(1)/Uhat  # does not converge without the Max(N,...)
        # self.tau_b = df.sqrt(tau_bx**2 + tau_by**2 + 1e-10)

        R_u_body = (- vi.intz(membrane_xx) - vi.intz(membrane_xy) - vi.intz(shear_xz) + tau_bx*lamda_x(1) - vi.intz(tau_dx))*df.dx(degree=3)
        R_v_body = (- vi.intz(membrane_yx) - vi.intz(membrane_yy) - vi.intz(shear_yz) + tau_by*lamda_y(1) - vi.intz(tau_dy))*df.dx(degree=3)

        self.coupler.R += R_u_body
        self.coupler.R += R_v_body

    def write_variables_pvd(self,t):
        # Us_temp = df.project(df.as_vector([self.u(0),self.v(0)]), self.Q)
        Us_temp = df.project(df.sqrt(self.u(0)**2+self.v(0)**2), self.coupler.Q_cg)
        # Ub_temp = df.project(df.as_vector([self.u(1),self.v(1)]), self.Q)
        Ub_temp = df.project(df.sqrt(self.u(1)**2+self.v(1)**2), self.coupler.Q_cg)

        self.Us.dat.data[:] = Us_temp.dat.data_ro
        self.Ub.dat.data[:] = Ub_temp.dat.data_ro
        self.Us_file.write(self.Us, time=t)
        self.Ub_file.write(self.Ub, time=t)


class GLADS(object):
    def __init__(self,mesh, results_dir):
        self.mesh     = mesh
        E_phi         = df.FiniteElement("CG", mesh.ufl_cell(), 1)
        E_h           = df.FiniteElement("DG", mesh.ufl_cell(),0)
        E_S           = df.FiniteElement("DGT", mesh.ufl_cell(),0)
        self.elements = [E_phi,E_h,E_S]
        # functions for expressions to save
        self.V_cg_vec = df.VectorFunctionSpace(self.mesh, "CG", 1)

        self.V_phi    = df.FunctionSpace(mesh,E_phi)
        self.V_h      = df.FunctionSpace(mesh,E_h)
        self.V_S      = df.FunctionSpace(mesh,E_S)

        # create output files
        self.results_dir = results_dir
        self.outfile_phi = df.VTKFile(results_dir+'df_phi.pvd')
        self.outfile_N   = df.VTKFile(results_dir+'df_N.pvd')
        self.outfile_h   = df.VTKFile(results_dir+'df_h.pvd')
        self.outfile_q   = df.VTKFile(results_dir+'df_q.pvd')
        self.outfile_S   = df.VTKFile(results_dir+'df_S.pvd')
        self.outfile_Q   = df.VTKFile(results_dir+'df_Q.pvd')

        # DG0 submesh function for saving channels (direct writing of DGT0 to pvd not supported at this point)
        self.S_save, self.S_submesh = hlp.make_subDG0(mesh)
        self.Q_save, self.Q_submesh = hlp.make_subDG0(mesh)

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

    def build_forms(self, m, e_v=1e-4, dt0=3600*2, k_s=0.05, k_c=0.5, h_r=0.5, l_r=5, p_s=0, l_c=10, alpha=1.25, beta=1.5, p=1, q=1, beta2=140,Uhat=1,Nhat=1, transition=False, omega=1/2000, u_b=30):
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
        phi0.interpolate(0.8*rho_i * g * H + rho_w * g * B)
        h0   = self.h0   = df.Function(self.V_h)
        h0.dat.data[:]   = 0.1*h_r
        S0   = self.S0   = df.Function(self.V_S)
        S0.dat.data[:]   = 0.001

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
        Q   = self.Q = -k_c*df.max_value(S,1e-15)**alpha*df.max_value(dphids**2, 1e-15)**(beta/2.-1)*dphids
        q_c = -k_s*df.max_value(h,1e-15)**alpha*df.max_value(dphids**2, 1e-15)**(beta/2.-1)*dphids

        # sheet flux
        if transition:
            # Tim Hill's transition model laminar/turbulent
            gradphi = df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(1/2)
            self.q_s = q_s = -nu/(2*omega) * (h_r/h)**(3-2*alpha) * (-1+df.sqrt(1+4*omega/nu*(h/h_r)**(3-2*alpha)*k_s*h**3*gradphi))*df.grad(phi)/gradphi
        else:
            # 'original' GlaDS
            self.q_s = q_s   = -k_s*df.max_value(h,1e-15)**alpha*df.max_value(df.dot(df.grad(phi),df.grad(phi)),1e-15)**(beta/2.-1)*df.grad(phi)
        # save Reynolds number for visualization
        self.Re = q_s/nu

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

        # sliding speed
        if hasattr(self.coupler, 'stokes'):  # from ice flow model if coupled
            u_b = df.sqrt(self.coupler.stokes.u(1)**2 + self.coupler.stokes.v(1)**2 + 1e-10)
        else:                                # prescribed if hydro only
            u_b = df.Constant(u_b)

        # melt through frictional heat
        # tau_b = -beta2*(Max(N,1e4)/Nhat)**p*(u_b/Uhat)**(q-1)*u_b/Uhat
        # M_fr = self.M_fr = abs(tau_b*u_b)/(rho_i*L)

        # log-normal bump size
        # sigma_hr = 1
        # log_h_r_s = np.array([hh for hh in np.log(h_r(0)) + sigma_hr*np.linspace(-3,3,15)])
        # probs = np.exp(-0.5*(log_h_r_s - np.log(h_r(0)))**2/(sigma_hr)**2)
        # probs/=probs.sum()
        # h_r_s = [df.Constant(np.exp(lhr)) for lhr in log_h_r_s]

        # opening and closure for sheets and channels
        O   = df.max_value(u_b*(h_r - h)/l_r,0)
        # O   = df.max_value(u_b*(h_r - h)**(p_s+1)/l_r / h_r**p_s,0)
        # O = sum([p*(h_r/l_r)*u_b*Max(1 - h/h_r_i,0) for p,h_r_i in zip(probs,h_r_s)])
        C   = A*h*abs(N)**(n-1)*N
        O_c = (Chi-Pi) / (rho_i*L)
        C_c = A*S*abs(N)**(n-1)*N

        R_phi_h = (xsi*e_v/(rho_w*g)*(phi-phi0)/dt  - df.dot(df.grad(xsi),q_s) + xsi * (O-C-m) ) * df.dx(degree=3)
        R_phi_S = df.avg(-dxsids*Q + xsi * O_c*(1-rho_i/rho_w) - xsi * C_c) * df.dS
        R_h     = ((h - h0)/dt - O + C) * psi * df.dx(degree=3)
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
        self.outfile_N.write(df.project(self.N, self.V_phi, name="N"), time=t)
        self.outfile_h.write(df.project(self.coupler.U.sub(1), self.V_h, name="h"), time=t)
        self.outfile_q.write(df.project(self.q_s, self.V_cg_vec, name="q"), time=t)
        hlp.save_DGT0(self.mesh, self.S_submesh, self.coupler.U.sub(2), self.S_save, self.outfile_S, t)
        hlp.save_DGT0(self.mesh, self.Q_submesh, df.project(abs(self.Q),self.V_S), self.Q_save, self.outfile_Q, t)

    def save_end_state(self, chk_file, csv_file):
        with CheckpointFile(chk_file, 'w') as afile:
            afile.save_mesh(self.mesh)  # optional
            afile.save_function(self.coupler.U.sub(0), name="phi")
            afile.save_function(self.coupler.U.sub(1), name="h")
        df_S = pd.DataFrame({'S': self.coupler.U.sub(2).dat.data_ro})
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
            self.S0.dat.data[:] = S0_
        else:
            self.S0.interpolate(S0_)
        self.coupler.U.sub(2).assign(self.S0)


class Coupler(object): # ice flow + hydro
    def __init__(self,mesh,stokes,hydro):

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

        self.stokes = stokes
        self.hydro  = hydro

        elements = self.hydro.elements + self.stokes.elements

        self.E_V      = df.MixedElement(elements)
        self.V   = df.FunctionSpace(mesh,self.E_V)

        E_cg      = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.Q_cg = df.FunctionSpace(mesh,E_cg)

        self.U      = df.Function(self.V)
        self.Lambda = df.TestFunction(self.V)  # or Function??

        self.R = 0

    def set_geometry(self,B,H):
        self.B = df.Function(self.hydro.V_phi).project(B)
        self.H = df.Function(self.hydro.V_phi).project(H)
        self.S = self.B + self.H


class Coupler(object):  # hydro only
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

        self.E_V      = df.MixedElement(elements)
        self.V   = df.FunctionSpace(mesh,self.E_V)

        E_cg      = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.Q_cg = df.FunctionSpace(mesh,E_cg)

        self.U      = df.Function(self.V)
        self.Lambda = df.TestFunction(self.V)  # or Function??

        self.R = 0

    def set_geometry(self,B,H):
        self.B = df.Function(self.hydro.V_phi).project(B)
        self.H = df.Function(self.hydro.V_phi).project(H)
        self.S = self.B + self.H
