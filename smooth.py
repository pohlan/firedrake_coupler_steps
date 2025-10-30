import os
import firedrake as df
from firedrake.__future__ import interpolate
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor, triplot
import xarray as xr
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from firedrake.checkpointing import CheckpointFile
from datetime import datetime, timedelta
import re

def load_and_smooth(name, sig=8):
    min_x = -2.5e5
    min_y = -2.675e6
    max_x = 6e4
    max_y = -2.450e6

    r = gu.Raster(f"NETCDF:Greenland_data/BedMachineGreenland-v5.nc:{name}")
    r.crop((min_x,min_y,max_x,max_y), inplace=True)

    dat_new = gaussian_filter(r.data.data, sigma=sig)
    r_new = gu.Raster.from_array(data=dat_new, transform=r.transform, crs=r.crs)
    r_new.save(f"Greenland_data/BedMachineGreenland-v5_{name}_smooth_sig{sig}.nc")

load_and_smooth("bed")
load_and_smooth("thickness")
