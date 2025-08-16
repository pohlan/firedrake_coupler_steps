import firedrake as df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, triplot
from firedrake.checkpointing import CheckpointFile
import pyvista as pv


# A = pv.read('/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_mid_resolution/Bed/B_c.pvd')[0]
A = pv.read('/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_low_resolution/Bed/B_c.pvd')[0]
x_csv = np.array(A.points[:,0])
y_csv = np.array(A.points[:,1])
z_B = np.array(A.active_scalars)
# A = pv.read('/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_mid_resolution/Thk/H_c.pvd')[0]
A = pv.read('/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_low_resolution/Thk/H_c.pvd')[0]
z_H = np.array(A.active_scalars)

# csv_file = '/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/step_10b/data/BH_vec.csv'
# x_csv    = np.float64(pd.read_csv(csv_file).x)
# y_csv    = np.float64(pd.read_csv(csv_file).y)
# z_B      = np.float64(pd.read_csv(csv_file).z_B)
# z_H      = np.float64(pd.read_csv(csv_file).z_H)

# mesh = df.Mesh('step_10b/data/western_med_v2.msh')
mesh = df.Mesh('step_10b/data/western_low_v2.msh')
x_df = mesh.coordinates.dat.data[:,0]
y_df = mesh.coordinates.dat.data[:,1]

# zB_df = np.zeros(len(x_df))
# zH_df = np.zeros(len(x_df))
# idx = np.zeros(len(x_csv), dtype=int)
# for (n,(xx,yy)) in enumerate(zip(x_csv, y_csv)):
#     idx[n] = np.where(mesh.coordinates.dat.data == [xx, yy])[0][0]
#     zB_df[idx[n]] = z_B[n]
#     zH_df[idx[n]] = z_H[n]

idx = np.zeros(len(x_csv), dtype=int)
for (n,(xx,yy)) in enumerate(zip(x_df, y_df)):
    idx[n] = np.where((x_csv == xx) & (y_csv == yy))[0][0]
zB_df = z_B[idx]
zH_df = z_H[idx]

V = df.FunctionSpace(mesh, 'CG', 1)
B = df.Function(V)
B.vector()[:] = zB_df
H = df.Function(V)
H.vector()[:] = zH_df




##############

plt.figure()
plt.scatter(x_csv, y_csv, 3, z_B)
plt.xlim(-2.2e5, -1.8e5)
plt.ylim(-2.55e6, -2.5e6)
plt.savefig("z_B_scatter1.jpg")

plt.figure()
plt.scatter(x_df, y_df, 3, zB_df)
plt.xlim(-2.2e5, -1.8e5)
plt.ylim(-2.55e6, -2.5e6)
plt.savefig("z_B_scatter2.jpg")

fig, axes = plt.subplots()
cl = tripcolor(B, axes=axes)
fig.colorbar(cl)
plt.xlim(-2.2e5, -1.8e5)
plt.ylim(-2.55e6, -2.5e6)
# plt.axis('equal')
plt.savefig("z_B.jpg")
fig, axes = plt.subplots()
cl = tripcolor(H, axes=axes)
fig.colorbar(cl)
plt.savefig("z_H.jpg")

chk_file = 'step_10b/data/fenics_geom.h5'
with CheckpointFile(chk_file, 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(B, name="B")
    afile.save_function(H, name="H")
