import numpy as np
import xarray as xr
import gstools as gs
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# # 1) grid
# Lx, Ly = 100*10**3, 25*10**3
# nx, ny = 200, 50
# x = np.linspace(0.0, Lx, nx)
# y = np.linspace(0.0, Ly, ny)

# # 2) covariance model (exponential)
# var = (1)**2
# mean = -3.6
# len_scale = 4000
# model = gs.Exponential(dim=2, var=var, len_scale=len_scale)

# # 3) random generator
# # seed = 1234
# srf = gs.SRF(model, mean=mean) #, seed=seed)

# # 4) generate field (on regular coords)
# field = srf((x, y), mesh_type="structured")
# # field shape: (nx, ny)

# # 5) store as xarray and to netcdf
# da = xr.DataArray(
#     data=10**(field),
#     dims=("x", "y"),
#     coords={"x": x, "y": y},
#     attrs={
#         "long_name": "Gaussian random field",
#         "covariance": f"Exponential(var={var}, len_scale={len_scale})"
#     },
# )
# ds = xr.Dataset(
#     {"field": da},
#     attrs={
#         "title": "Random field with gstools",
#         "history": f"generated with gstools",
#     },
# )
# outfn = "random_field_r4k_py.nc"
# ds.to_netcdf(outfn, format="NETCDF4")
# print("Saved", outfn)

# # Plot the random field
# fig, ax = plt.subplots(figsize=(12, 4))
# im = ax.contourf(x / 1e3, y / 1e3, 10**(field.T), levels=20, cmap='viridis', norm=colors.LogNorm(vmin=1e-6, vmax=0.03))
# ax.set_xlabel('x (km)')
# ax.set_ylabel('y (km)')
# ax.set_title('Gaussian Random Field')
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('Field value')
# fig.tight_layout()
# fig.savefig('random_field.jpg', dpi=150, bbox_inches='tight')
# print("Saved random_field.jpg")


import firedrake as df
from firedrake.checkpointing import CheckpointFile
import models_main.helpers as hlp

# 1) grid
mesh = df.Mesh("Greenland_data/russel/russel.msh")
x, y = zip(*hlp.get_coordinates(mesh, "CG", 1))

# 2) covariance model (exponential)
# var = 0.003**2
# mean = 0.002
var = (1)**2
mean = -3.6
len_scale = 3000
model = gs.Exponential(dim=2, var=var, len_scale=len_scale)

# 3) random generator
seed = 1238
# seed = 1239
# seed = 1245
srf = gs.SRF(model, mean=mean, seed=seed)

# 4) generate field (on regular coords)
field = srf((x, y), mesh_type="unstructured")
# field shape: (nx, ny)

# 5) plot the random field
V = df.FunctionSpace(mesh, "CG", 1)
ks = df.Function(V)
ks.dat.data[:] = np.maximum(10**field, 1e-5)

from firedrake.pyplot import tripcolor, triplot


fig, axes = plt.subplots()
cl = tripcolor(ks, axes=axes,  norm=colors.LogNorm(vmin=1e-5, vmax=0.01))
axes.set_aspect("equal")
fig.colorbar(cl)
plt.savefig("ks.jpg")

df.VTKFile("ks.pvd").write(ks)

# 6) Save
chk_file = "ks_russel.h5"
with CheckpointFile(chk_file, 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(ks, name="k_s")

