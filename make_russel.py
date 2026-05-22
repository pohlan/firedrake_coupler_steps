import gmsh
import numpy as np
import fiona
import matplotlib.pyplot as plt
import geopandas as gpd

gmsh.initialize()
gmsh.model.add("domain")
geometry = gmsh.model.geo

data_dir = "/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/Greenland_data/russel/"

gdf = gpd.read_file("Greenland_data/russel/russel_domain.gpkg")
geom = gdf.geometry[0] #.geoms[0]
coords_outer = geom.exterior.coords[:-1]  # last point is double, remove it
# if there is an actual hole in the polygon
# coords_inner = [ring.coords[:-1] for ring in geom.interiors][0]  # for several holes, remove zero and ajust accordingly in code below

def build_ring(coords):
    pts = []
    for x, y in coords:
        pts.append(geometry.addPoint(x, y, 0))

    lines = []
    for i in range(len(pts)-1):
        lines.append(geometry.addLine(pts[i], pts[i+1]))
    lines.append(geometry.addLine(pts[-1], pts[0]))

    return geometry.addCurveLoop(lines), lines

outer_loop, outer_lines = build_ring(coords_outer)
# inner_loop, inner_lines = build_ring(coords_inner)

# plane = geometry.addPlaneSurface([outer_loop, inner_ loop])
plane = geometry.addPlaneSurface([outer_loop])

gmsh.model.geo.synchronize()


xpts = [c[0] for c in coords_outer]
ypts = [c[1] for c in coords_outer]
# xpts = np.concatenate([ [c[0] for c in coords_outer], [c[0] for c in coords_inner]])
# ypts = np.concatenate([ [c[1] for c in coords_outer], [c[1] for c in coords_inner]])
# lines = np.concatenate([outer_lines, inner_lines])
lines = np.concatenate([outer_lines])

# label boundary next to hydrological outlet points
# get coordinates
hydro_file = f"{data_dir}hydrological_outlets_russel.gpkg"
hydro_points = fiona.open(hydro_file, mode="r")
hydro_coords = np.array([i['geometry']['coordinates'] for i in list(hydro_points.values())])
hydro_points.close()
# find closest boundary edges
n_lines = 1  # per outlet
i_bc_lines = np.zeros(len(hydro_coords)*n_lines, dtype=int)
for (k,crd) in enumerate(hydro_coords):
    i_pts = np.argpartition(np.sqrt((crd[0]-xpts)**2+(crd[1]-ypts)**2), n_lines+1)[:(n_lines+1)]
    i_pts.sort(axis=0)
    i_bc_lines[k*n_lines:(k*n_lines+n_lines)] = i_pts[0:n_lines]
# label
physical_line = gmsh.model.addPhysicalGroup(1, np.array(lines)[i_bc_lines])
non_bc_lines = np.delete(lines, i_bc_lines, axis=0)
physical_line = gmsh.model.addPhysicalGroup(1, non_bc_lines)
physical_surface = gmsh.model.addPhysicalGroup(2, [plane])


# Set triangle sizes
f = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(f, "F", "300 + 20000 / (1+exp(-8e-5*(x+180e3)))")
# constant option
# gmsh.model.mesh.field.setString(f, "F", "5000")
gmsh.model.mesh.field.setAsBackgroundMesh(f)

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
