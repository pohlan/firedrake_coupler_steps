import firedrake as df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, triplot
from firedrake.checkpointing import CheckpointFile
from firedrake.output import VTKFile
import pyvista as pv

mesh = df.Mesh("Greenland_data/russel/western_med_v2.msh")
x_df = mesh.coordinates.dat.data[:,0]
y_df = mesh.coordinates.dat.data[:,1]
V = df.FunctionSpace(mesh, 'CG', 1)

# for bed and thickness

# A = pv.read('/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_mid_resolution/Bed/B_c.pvd')[0]
# A = pv.read('/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_low_resolution/Bed/B_c.pvd')[0]
# x_csv = np.array(A.points[:,0])
# y_csv = np.array(A.points[:,1])
# z_B = np.array(A.active_scalars)
# A = pv.read('/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_mid_resolution/Thk/H_c.pvd')[0]
# A = pv.read('/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_low_resolution/Thk/H_c.pvd')[0]
# z_H = np.array(A.active_scalars)

# csv_file = '/home/annegret/Projects/coupled_modeling/firedrake_coupler_steps/step_10b/data/BH_vec.csv'
# x_csv    = np.float64(pd.read_csv(csv_file).x)
# y_csv    = np.float64(pd.read_csv(csv_file).y)
# z_B      = np.float64(pd.read_csv(csv_file).z_B)
# z_H      = np.float64(pd.read_csv(csv_file).z_H)

# ... (missing some lines)

##############

# plt.figure()
# plt.scatter(x_csv, y_csv, 3, z_B)
# plt.xlim(-2.2e5, -1.8e5)
# plt.ylim(-2.55e6, -2.5e6)
# plt.savefig("z_B_scatter1.jpg")

# plt.figure()
# plt.scatter(x_df, y_df, 3, zB_df)
# plt.xlim(-2.2e5, -1.8e5)
# plt.ylim(-2.55e6, -2.5e6)
# plt.savefig("z_B_scatter2.jpg")

# fig, axes = plt.subplots()
# cl = tripcolor(B, axes=axes)
# fig.colorbar(cl)
# plt.xlim(-2.2e5, -1.8e5)
# plt.ylim(-2.55e6, -2.5e6)
# # plt.axis('equal')
# plt.savefig("z_B.jpg")
# fig, axes = plt.subplots()
# cl = tripcolor(H, axes=axes)
# fig.colorbar(cl)
# plt.savefig("z_H.jpg")



# for SMB

f_VTK = VTKFile("Greenland_data/russel/SMB_fenics/SMB.pvd")
SMB = df.Function(V)

for i in range(12):
    file    = f"/home/annegret/Projects/coupled_modeling/glads_hybrid_demo/data_mid_resolution/SMB/smb_{i}_c.pvd"
    A       = pv.read(file)[0]
    x_csv   = np.array(A.points[:,0])
    y_csv   = np.array(A.points[:,1])
    smb_csv = np.array(A.active_scalars)

    idx = np.zeros(len(x_csv), dtype=int)
    for (n,(xx,yy)) in enumerate(zip(x_df, y_df)):
        idx[n] = np.where((x_csv == xx) & (y_csv == yy))[0][0]
    smb_df = smb_csv[idx]

    SMB.vector()[:] = smb_df
    f_VTK.write(SMB)

    chk_file = f"Greenland_data/russel/SMB_fenics/SMB_{i}.h5"
    with CheckpointFile(chk_file, 'w') as afile:
        afile.save_mesh(mesh)
        afile.save_function(SMB, name="SMB")
