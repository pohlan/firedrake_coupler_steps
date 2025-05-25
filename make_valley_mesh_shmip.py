import gmsh
import numpy as np

# outline
para_bench = 0.05
def surface(x,y):
    return 100*(x+200)**(1/4) + 1/60*x - 2e10**(1/4) + 1
def f(x,para):
    return (surface(6e3,0) - para*6e3)/6e3**2 * x**2 + para*x
def g(y):
    return 0.5e-6 * abs(y)**3
def h(x,para):
    return (-4.5*x/6e3 + 5) * (surface(x,0)-f(x, para)) / (surface(x,0)-f(x, para_bench)+1e-15)
def ginv(x): # the inverse of g
    return (x/0.5e-6)**(1/3)
def outline(x):
    return ginv( (surface(x,0)-f(x,0.05))/(h(x,0.05)+1e-15) )

# evaluate on a number of points
x = np.concatenate([np.arange(0,450,150), np.arange(450, 6e3, 250)])
y = outline(x)
xpts = np.concatenate([x,np.flip(x), [0]])
ypts = np.concatenate([y,np.flip(-y), [0]])
# xpts = np.concatenate([x, [6e3], np.flip(x), [0.]])
# ypts = np.concatenate([y, [0], np.flip(-y), [0.]])
# solver doesn't like the point 6e3, 0 for some reason, maybe because H=0 there.

# generate mesh with gmsh
gmsh.initialize()
geometry = gmsh.model.geo

lc  = 220
points = [geometry.add_point(xi,yi,0,lc) for (xi,yi) in zip(xpts,ypts)]
lines  = [geometry.add_line(pt1, pt2) for (pt1,pt2) in zip(points, np.concatenate([points[1:],[points[0]]])) ]

face  = geometry.add_curve_loop(lines)
plane = geometry.add_plane_surface([face])
physical_line = geometry.add_physical_group(1, [lines[-1],lines[-2]]) # bc edge
physical_surface = geometry.add_physical_group(2, [plane])

geometry.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("valley.msh")
gmsh.finalize()

# plot
import firedrake as df
from firedrake.pyplot import tripcolor, triplot
import matplotlib.pyplot as plt
mesh = df.Mesh("valley.msh")
fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.legend()
axes.axis("equal")
plt.savefig("valley_mesh.jpg")

# print how many dofs
V_phi = df.FunctionSpace(mesh, "CG", 1)
V_h   = df.FunctionSpace(mesh, "CG", 1)
V_S   = df.FunctionSpace(mesh, "DGT", 0)
print(V_phi.dof_count)
print(V_h.dof_count)
print(V_S.dof_count)
