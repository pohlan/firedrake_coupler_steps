import firedrake as df
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import geoutils as gu

def get_catchments_russel(mesh):
    x,y  = df.SpatialCoordinate(mesh)
    DG0 = df.FunctionSpace(mesh, "DG", 0)
    mesh_xs = df.Function(DG0).interpolate(x)
    mesh_ys = df.Function(DG0).interpolate(y)
    mesh_pts_list = gpd.points_from_xy(x=mesh_xs.dat.data_ro, y=mesh_ys.dat.data_ro, crs=3413)

    S = df.Function(DG0)
    r_surf = gu.Raster(f"NETCDF:Greenland_data/BedMachineGreenland-v5.nc:surface")
    S.dat.data[:] = r_surf.interp_points((mesh_xs.dat.data_ro, mesh_ys.dat.data_ro), as_array=True)

    catchments = gpd.read_file("Greenland_data/Yang_Smith_2016/ds01/catchments_LC80070132013231LGN00_B8.shp").geometry
    additional_catchments = gpd.read_file("Greenland_data/Yang_Smith_2016/additional_catchments.gpkg").geometry  # manually outlined, where there were moulins/rivers but no catchment
    moulins_gdf = gpd.read_file("Greenland_data/Yang_Smith_2016/ds01/moulins_LC80070132013231LGN00_B8.shp").geometry
    moulins = moulins_gdf.geometry
    lakes = gpd.read_file("Greenland_data/Yang_Smith_2016/ds01/lakes_LC80070132013231LGN00_B8.shp").geometry
    assert moulins.crs == catchments.crs
    assert lakes.crs == catchments.crs

    # get lake elevations
    lakes_3413 = moulins_gdf.to_crs(3413).geometry  # need the other crs to interpolate surface elevation
    x_lakes = [p.x for p in lakes_3413]
    y_lakes = [p.y for p in lakes_3413]
    S_lakes = r_surf.interp_points((x_lakes, y_lakes), as_array=True)

    # merge catchments with additional manually drawn catchments
    # assert catchments.crs == additional_catchments.crs
    # catchments = gpd.GeoDataFrame(pd.concat([catchments, additional_catchments], ignore_index=True), crs=catchments.crs).geometry

    # change crs of mesh points to match catchment/moulin/lake crs
    mesh_pts = gpd.GeoDataFrame(geometry=mesh_pts_list, crs=3413)
    mesh_pts.to_crs(catchments.crs, inplace=True)

    # get model domain
    model_domain = gpd.read_file("Greenland_data/russel/russel_domain.gpkg")
    model_domain.to_crs(catchments.crs, inplace=True)
    model_domain = model_domain.geometry[0]

    input_loc = []
    i_catchments_glob = []
    facet_functions = []
    catchment_fct = df.Function(DG0)
    n_lakes = 0

    def get_facet_f(ch):
        facet_f = df.Function(DG0)
        i_mesh = np.where(mesh_pts.geometry.within(ch))
        facet_f.dat.data[i_mesh] = 1
        return facet_f, i_mesh

    i_catch_domain = 1  # starting from 1 because zero means no catchment
    for (i_ch,ch) in enumerate(catchments):
        facet_f, i_mesh = get_facet_f(ch)
        if len(i_mesh[0]) == 0:  # outside of model domain
            continue
        # find moulin index of catchment (if any)
        i = np.where(moulins.within(ch) & moulins.within(model_domain))[0]
        if i.size == 0:
            # no moulin, search for lakes
            i_lakes = np.where([l.within(ch) & l.within(model_domain) for l in lakes])[0]
            if i_lakes.size == 0:
                continue
            imin    = np.argmin(S_lakes[i_lakes])  # take lake at lowest elevation if there are several
            # save centroid coordinate of that lake polygon
            input_loc.append(lakes[i_lakes[imin]].centroid)
            # update the catchment function
            catchment_fct.dat.data[i_mesh] = i_catch_domain # just to track which elements have been assigned, and to visualize
            facet_functions.append(facet_f)
            i_catchments_glob.append(i_ch)
            n_lakes += 1
            i_catch_domain += 1
        else:
            # copy moulin coordinates to input_loc
            input_loc.append(moulins[i[0]])
            # delete it from m_points to keep track of which moulins have been assigned yet
            # moulins.pop(i[0])
            # moulins.reset_index(drop=True, inplace=True)  # otherwise i wouldn't agree with index of list anymore
            # update the catchment function
            catchment_fct.dat.data[i_mesh] = i_catch_domain
            facet_functions.append(facet_f)
            i_catchments_glob.append(i_ch)
            i_catch_domain += 1

    # check that each input location is within the respective catchment
    assert len(facet_functions) == len(input_loc)
    for (input, ch) in zip(input_loc, catchments[i_catchments_glob]):
        assert input.within(ch)

    # save catchment partition on mesh
    df.VTKFile('Greenland_data/russel/catchments_russel10.pvd').write(facet_functions[10])
    df.VTKFile('Greenland_data/russel/catchments_russel30.pvd').write(facet_functions[30])
    df.VTKFile('Greenland_data/russel/catchments_russel99.pvd').write(facet_functions[99])
    df.VTKFile('Greenland_data/russel/catchments_russel.pvd').write(catchment_fct)

    # save input locations in dataframe
    # (convert back to mesh CRS (EPSG:3413) so these can be used directly with mesh coordinates)
    input_loc_gs = gpd.GeoSeries(input_loc, crs=catchments.crs).to_crs(3413)
    df_moul = pd.DataFrame({"x": [p.x for p in input_loc_gs], "y": [p.y for p in input_loc_gs]})

    # save distributed melt mask where there are no catchments --> distributed melt forcing
    i_catchment = np.nonzero(catchment_fct.dat.data_ro)[0]
    mean_S = np.mean(S.dat.data_ro[i_catchment])
    i_distributed = np.where((catchment_fct.dat.data_ro == 0) & (S.dat.data_ro < mean_S))
    distributed_melt_mask = df.Function(df.Function(DG0)).interpolate(0)
    distributed_melt_mask.dat.data[i_distributed] = 1
    df.VTKFile('Greenland_data/russel/distributed_melt_mask.pvd').write(distributed_melt_mask)

    return df_moul, facet_functions, distributed_melt_mask

# other option: calculate moulins instead of using the manually drawn ones:

# # keep only moulins that are in the domain
# i_domain = np.where(moulins.within(model_domain))[0]
# moulins = moulins[i_domain]
# moulins.reset_index(drop=True, inplace=True)
# moulins_id = np.arange(len(moulins))+len(input_loc)

# # moulin locations and surface elevation
# moulins_3413 = moulins_gdf.to_crs(3413).geometry[i_domain]  # need the other crs to interpolate surface elevation
# x_moulins = [p.x for p in moulins_3413]
# y_moulins = [p.y for p in moulins_3413]
# S_moulins = r_surf.interp_points((x_moulins, y_moulins), as_array=True)

# # extend facet_functions
# facet_functions.extend([df.Function(DG0) for i in range(len(moulins))])

# # loop through mesh coords and assign them to a moulin they drain to
# for (i_mesh,msh) in enumerate(mesh_pts.geometry):
#     if catchment_fct.dat.data[i_mesh] != 0:  # element has alrady been assigned to a catchment
#         continue
#     dists = msh.distance(moulins)
#     i_downstream = np.where((S_moulins -10 < S.dat.data_ro[i_mesh]) & (dists < 5000))[0] # which moulins are downstream of this location
#     if i_downstream.size == 0:
#         continue

#     i_min = np.argmin(dists[i_downstream])
#     catchment_fct.dat.data[i_mesh] = moulins_id[i_downstream[i_min]]
#     facet_functions[moulins_id[i_downstream[i_min]]].dat.data[i_mesh] = 1

# input_loc = np.concatenate([input_loc,moulins])



# mesh_c = df.RelabeledMesh(mesh, facet_functions, list(range(len(facet_functions))))

# df.VTKFile('catchments_russel.pvd').write(facet_functions[0])

# print(f"n_lakes = {n_lakes}")
