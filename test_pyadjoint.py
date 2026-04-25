from firedrake.adjoint import minimize, ReducedFunctional, Control, continue_annotation
from firedrake import *
import numpy as np

n = 30
mesh = UnitSquareMesh(n, n)
timestep = Constant(5.0/n)
steps = 10

x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
ic = project(0.5 - sqrt((x-0.5)**2+(y-0.5)**2), V, name="ic")
VTKFile("m_ic.pvd").write(ic)

u_old = Function(V, name="u_old")
u_new = Function(V, name="u")
v = TestFunction(V)
u_old.assign(ic)
nu_true = project(0.08*x, V, name="nu_true")
VTKFile("m_nu.pvd").write(nu_true)
q = -nu_true*grad(u_new)
F = ((u_new-u_old)/timestep*v - inner(grad(v),q))*dx
bc = DirichletBC(V, 0.0, "on_boundary")
problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
solver = NonlinearVariationalSolver(problem)

ufile = VTKFile("m_sol.pvd")

u_obs = []
for n in range(steps):
    solver.solve()
    ufile.write(u_new, time=n)
    u_old.assign(u_new)
    u_obs.append(u_new.copy(deepcopy=True))


### reset ###
continue_annotation()


m = Function(V, name="nu exp")
m.interpolate(-4)
nu = exp(m)

u_new = Function(V, name="u")
u_old = Function(V, name="u_old")
u_old.assign(ic)

# timestep = 0.5*h**2/max_value(nu)
q = -nu*grad(u_new)
F = ((u_new-u_old)/timestep*v - inner(grad(v),q))*dx
problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
solver = NonlinearVariationalSolver(problem)

J = 0

for n in range(steps):
    solver.solve()
    u_old.assign(u_new)
    # calculate misfit
    diff = u_new - u_obs[n]
    J += inner(diff,diff)*dx
    # print(assemble(J))

alpha = Constant(1e-4)
J += alpha * inner(grad(m),grad(m)) * dx
# print(assemble(alpha*inner(grad(m),grad(m)) * dx))

J = assemble(J)
Jhat = ReducedFunctional(J, Control(m))
print("Initial J:", Jhat(m))

# from firedrake.adjoint import taylor_test

# h = Function(V)
# h.interpolate(1.0)

# rate = taylor_test(Jhat, m, h)
# print("Taylor test rate:", rate)

result = minimize(Jhat, method="CG",
                #   bounds=(-15,-1.0),
                #   callback=cb,
                 options={"disp": True})

result_nu = project(exp(result), V, name="nu_opt")
VTKFile("m_opt.pvd").write(result_nu)

print(f"sum(nu - nu_true) integrated: {assemble((result_nu-nu_true)*dx)}")
