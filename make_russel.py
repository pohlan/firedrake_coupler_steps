import gmsh
import numpy as np
import fiona
import matplotlib.pyplot as plt

data_dir = '/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/step_8/data/'

# shp = fiona.open(f"{data_dir}russel_domain.gpkg", mode="r")
shp = fiona.open(f"{data_dir}russel_domain_large.gpkg", mode="r")
coords = np.array(list(shp.values())[0]['geometry']['coordinates']).squeeze()
shp.close()

# plt.scatter(coords[:,0],coords[:,1])

# len_scale = 108000
len_scale = 1


# Trim last point (is identical to first)
coords = coords[:-1, :]
xpts = coords[:,0] / len_scale
ypts = coords[:,1] / len_scale

print(len(ypts))

# generate mesh with gmsh
gmsh.initialize()
geometry = gmsh.model.geo

# lc  = 1200 / len_scale
lc  = 30000 / len_scale
points = [geometry.add_point(xi,yi,0,lc) for (xi,yi) in zip(xpts,ypts)]
lines  = [geometry.add_line(pt1, pt2) for (pt1,pt2) in zip(points, np.concatenate([points[1:],[points[0]]])) ]

face  = geometry.add_curve_loop(lines)
plane = geometry.add_plane_surface([face])

# 'small'
# physical_line = geometry.add_physical_group(1, np.concatenate([lines[-47:-40], lines[-28:-14],lines[10:20]]))
# physical_line = geometry.add_physical_group(1, np.concatenate([lines[0:10], lines[20:-47],lines[-40:-28],lines[-14:]]))

# 'large'
physical_line = geometry.add_physical_group(1, np.concatenate([lines[-47:-40], lines[-28:-14], lines[10:20], lines[170:180], lines[230:240]]))
physical_line = geometry.add_physical_group(1, np.concatenate([lines[0:10], lines[20:170],lines[180:230],lines[240:-47],lines[-40:-28],lines[-14:]]))

physical_surface = geometry.add_physical_group(2, [plane])

geometry.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write(f"{data_dir}russel.msh")
gmsh.finalize()

# plot
import firedrake as df
from firedrake.pyplot import tripcolor, triplot
import matplotlib.pyplot as plt
mesh = df.Mesh(f"{data_dir}russel.msh")
fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.legend()
axes.axis("equal")
plt.savefig(f"{data_dir}russel_mesh.jpg", dpi=300)
