import firedrake as df
from firedrake.__future__ import interpolate
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor, triplot
import xarray as xr
import scipy.interpolate as itp
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_geometry(B, H, mesh):
    # bed
    fig, axes = plt.subplots()
    cl = tripcolor(B, axes=axes)
    fig.colorbar(cl)
    plt.savefig("B.jpg")
    # thickness
    fig, axes = plt.subplots()
    cl = tripcolor(H, axes=axes)
    fig.colorbar(cl)
    plt.savefig("H.jpg")
    # mesh
    fig, axes = plt.subplots()
    colors = triplot(mesh, axes=axes)
    axes.legend()
    plt.savefig("mesh.jpg")

def scatterplt_fields(hydro, dest_id):
    for (field, name, element) in zip(hydro.U.subfunctions, ["phi", "h", "S"], hydro.E_V.sub_elements): # [hydro.E_phi, hydro.E_h, hydro.E_S]):
        # get dof coordinates
        Vmesh = df.VectorFunctionSpace(hydro.mesh, element)
        X = df.assemble(interpolate(hydro.mesh.coordinates,Vmesh))
        meshx = X.dat.data[:,0]
        meshy = X.dat.data[:,1]
        # scatter plot
        plt.figure()
        plt.scatter(meshx, meshy, 10, field.vector()[:])
        plt.colorbar()
        plt.savefig(hydro.results_dir+name+"_"+dest_id+".jpg")

def diff_to_glads_matlab(hydro, file):
    ds_mw = xr.open_dataset(file)
    x_mw = ds_mw.coords1.data[0]
    y_mw = ds_mw.coords1.data[1]
    # hard-codes that phi and h have the same dofs
    V_mesh = df.VectorFunctionSpace(hydro.mesh, hydro.E_V.sub_elements[0])
    X = df.assemble(interpolate(hydro.mesh.coordinates, V_mesh))
    meshx = X.dat.data[:,0]
    meshy = X.dat.data[:,1]

    F_mw = df.Function(hydro.V_phi)
    F_diff = df.Function(hydro.V_phi)
    for (f_mw, F_sol) in zip([ds_mw.N, ds_mw.h], [hydro.N, hydro.U.sub(1)]):
        F_mw.vector()[:] = itp.griddata((x_mw,y_mw), f_mw.data[0], (meshx,meshy), method="nearest")
        F_diff.interpolate(F_mw-F_sol)
        print(np.max(np.abs(F_diff.vector()[:]/F_mw.vector()[:])))
        # assert np.max(np.abs(F_diff.vector()[:]/F_mw.vector()[:])) < 0.05

    # also save GlaDS-matlab as pvd to visualize in paraview
    shmip_suit = os.path.basename(file)[0:2]
    F_mw.vector()[:] = itp.griddata((x_mw,y_mw), ds_mw.N.data[0], (meshx,meshy), method="nearest") # the default method, linear, gives nan values at the edges of the domain
    VTKFile(shmip_suit+"_glads-matlab_phi.pvd").write(df.project(910*9.8*hydro.H - F_mw + 1000*9.8*hydro.B, hydro.V_phi, name="phi"))
    F_mw.vector()[:] = itp.griddata((x_mw,y_mw), ds_mw.h.data[0], (meshx,meshy), method="nearest")
    VTKFile(shmip_suit+"_glads-matlab_h.pvd").write(df.project(F_mw, hydro.V_phi, name="h"))
