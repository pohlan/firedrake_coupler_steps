import numpy as np
import xarray as xr
import gstools as gs
import matplotlib.pyplot as plt

# 1) grid
Lx, Ly = 100*10**3, 25*10**3
nx, ny = 200, 50
x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)

# 2) covariance model (exponential)
var = 0.001**2
mean = 0.002
len_scale = 5000
model = gs.Exponential(dim=2, var=var, len_scale=len_scale)

# 3) random generator
# seed = 1234
srf = gs.SRF(model, mean=mean) #, seed=seed)

# 4) generate field (on regular coords)
field = srf((x, y), mesh_type="structured")
# field shape: (nx, ny)

# 5) store as xarray and to netcdf
da = xr.DataArray(
    data=field,
    dims=("x", "y"),
    coords={"x": x, "y": y},
    attrs={
        "long_name": "Gaussian random field",
        "covariance": f"Exponential(var={var}, len_scale={len_scale})"
    },
)
ds = xr.Dataset(
    {"field": da},
    attrs={
        "title": "Random field with gstools",
        "history": f"generated with gstools",
    },
)
outfn = "random_field_r5k_py.nc"
ds.to_netcdf(outfn, format="NETCDF4")
print("Saved", outfn)

# Plot the random field
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.contourf(x / 1e3, y / 1e3, field.T, levels=20, cmap='viridis')
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title('Gaussian Random Field')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Field value')
fig.tight_layout()
fig.savefig('random_field.jpg', dpi=150, bbox_inches='tight')
print("Saved random_field.jpg")
