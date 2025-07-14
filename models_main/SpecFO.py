import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
from firedrake.output import VTKFile

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
    def __init__(self,mesh):
        E_cg = self.E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.elements = [E_cg]*4

    def set_coupler(self,coupler):
        self.coupler = coupler

    def build_variables(self,n=3.0,g=9.81,rho_i=917.,rho_w=1000.,eps_reg=1e-5,Be=464158,p=1,q=1,beta2=2e-3,write_pvd=True,results_dir='./results/'):
        self.n = df.Constant(n)
        self.g = df.Constant(g)
        self.rho_i = df.Constant(rho_i)
        self.rho_w = df.Constant(rho_w)
        self.eps_reg = df.Constant(eps_reg)
        self.Be = df.Constant(Be)
        self.p = df.Constant(p)
        self.q = df.Constant(q)
        self.beta2 = df.Function(self.coupler.Q_r)
        self.beta2.vector()[:] = beta2

        self.ubar = self.coupler.U[0]
        self.vbar = self.coupler.U[1]
        self.udef = self.coupler.U[2]
        self.vdef = self.coupler.U[3]

        self.lamdabar_x = self.coupler.Lambda[0]
        self.lamdabar_y = self.coupler.Lambda[1]
        self.lamdadef_x = self.coupler.Lambda[2]
        self.lamdadef_y = self.coupler.Lambda[3]

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

        if write_pvd:
            self.write_pvd = True
            self.results_dir = results_dir
            # self.Us = df.project(df.as_vector([self.u(0),self.v(0)]), self.Q)
            self.Us = df.project(df.sqrt(self.u(0)**2+self.v(0)**2), self.coupler.Q_dg)
            # self.Ub = df.project(df.as_vector([self.u(1),self.v(1)]), self.Q)
            self.Ub = df.project(df.sqrt(self.u(1)**2+self.v(1)**2), self.coupler.Q_dg)
            self.Us_file = VTKFile(results_dir+'U_s.pvd')
            self.Ub_file = VTKFile(results_dir+'U_b.pvd')

    def write_variables(self,t):
        # Us_temp = df.project(df.as_vector([self.u(0),self.v(0)]), self.Q)
        Us_temp = df.project(df.sqrt(self.u(0)**2+self.v(0)**2), self.coupler.Q_dg)
        # Ub_temp = df.project(df.as_vector([self.u(1),self.v(1)]), self.Q)
        Ub_temp = df.project(df.sqrt(self.u(1)**2+self.v(1)**2), self.coupler.Q_dg)

        self.Us.vector().set_local(Us_temp.vector().get_local())
        self.Ub.vector().set_local(Ub_temp.vector().get_local())
        self.Us_file.write(self.Us, time=t)
        self.Ub_file.write(self.Ub, time=t)

    def build_forms(self):
        n = self.n
        g = self.g
        rho_i = self.rho_i
        rho_w = self.rho_w
        eps_reg = self.eps_reg
        Be = self.Be
        p = self.p
        q = self.q
        beta2 = self.beta2

        u = self.u
        v = self.v
        lamda_x = self.lamda_x
        lamda_y = self.lamda_y
        H = self.coupler.H_c
        B = self.coupler.B_c
        S = self.coupler.S
        # N = self.coupler.hydro.N
        N = 1.0*rho_i*g*H

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

        tau_bx = -beta2*Max(N,5e4)**p*abs(u(1)**2 + v(1)**2 + 1e-3)**((q-1)/2.)*u(1)
        tau_by = -beta2*Max(N,5e4)**p*abs(u(1)**2 + v(1)**2 + 1e-3)**((q-1)/2.)*v(1)

        R_u_body = (- vi.intz(membrane_xx) - vi.intz(membrane_xy) - vi.intz(shear_xz) + tau_bx*lamda_x(1) - vi.intz(tau_dx))*df.dx
        R_v_body = (- vi.intz(membrane_yx) - vi.intz(membrane_yy) - vi.intz(shear_yz) + tau_by*lamda_y(1) - vi.intz(tau_dy))*df.dx

        self.coupler.R += R_u_body
        self.coupler.R += R_v_body


class Coupler(object):
    def __init__(self,mesh,stokes): #,hydro):

        self.stokes = stokes
        # self.hydro  = hydro

        elements = self.stokes.elements #+ self.hydro.elements
        E_V      = df.MixedElement(elements)
        self.V   = df.FunctionSpace(mesh,E_V)

        E_cg      = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.Q_cg = df.FunctionSpace(mesh,E_cg)

        E_dg      = df.FiniteElement("DG",mesh.ufl_cell(),0)
        self.Q_dg = df.FunctionSpace(mesh,E_dg)

        E_r       = df.FiniteElement('R',mesh.ufl_cell(),0)
        self.Q_r  = df.FunctionSpace(mesh,E_r)
        self.dw   = df.TestFunction(self.Q_r)

        self.space_list = [df.FunctionSpace(mesh,E) for E in elements]  # switched E and mesh

        self.U      = df.Function(self.V)
        self.Lambda = df.TestFunction(self.V)  # or Function??

        self.R = 0

    def set_geometry(self,B_c,B_d,H_c,H_d):
        self.B_c = B_c
        self.H_c = H_c
        self.B_d = B_d
        self.H_d = H_d
        self.S = B_c + H_c

    # def set_forcing(self,m):
    #     self.m = m
