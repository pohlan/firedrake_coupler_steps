import os
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point
import geoutils as gu

import statsmodels.api as sm

df_L = pd.read_csv("Greenland_data/forcing/KAN_L_day.csv")
df_M = pd.read_csv("Greenland_data/forcing/KAN_M_day.csv")
df_U = pd.read_csv("Greenland_data/forcing/KAN_U_day.csv")

# extract time series, find overlapping period
time_L = pd.to_datetime(df_L.time)
time_M = pd.to_datetime(df_M.time)
time_U = pd.to_datetime(df_U.time)
t_0 = np.max([time_L.iloc[0],time_M.iloc[0],time_U.iloc[0]])
t_end = np.min([time_L.iloc[-1],time_M.iloc[-1],time_U.iloc[-1]])
# crop all time series to the overlapping time
df_L = df_L.iloc[np.where((time_L >= t_0) & (time_L <= t_end))]
df_M = df_M.iloc[np.where((time_M >= t_0) & (time_M <= t_end))]
df_U = df_U.iloc[np.where((time_U >= t_0) & (time_U <= t_end))]
time = time_L.iloc[np.where((time_L >= t_0) & (time_L <= t_end))]

# get array of how many days each point is away from 2016-01-01
i_0 = np.where(time == np.datetime64('2016-01-01'))
time_np = np.array(time)
dT_days = (time_np - time_np[i_0]) / np.timedelta64(1, 'D')

# convert to np array to avoi weird behavior
T_L = np.array(df_L.t_u)
z_L = np.array(df_L.alt)
T_M = np.array(df_M.t_u)
z_M = np.array(df_M.alt)
T_U = np.array(df_U.t_u)
z_U = np.array(df_U.alt)

# also make a smoother version with lowess filter
lowess = sm.nonparametric.lowess
T_L_smooth = lowess(T_L, time, frac=1/110, xvals=time)

# save as csv
data_dir = 'Greenland_data/'
dff = pd.DataFrame({'dT_days':dT_days, 'T_L': T_L, 'T_L_smooth':T_L_smooth, 'z_L': z_L})
dff.to_csv(data_dir+'KAN_melt.csv',index=False)

# calculate melt from temperature to visualize
f_m = 0.01*365
melt = f_m*np.array([np.max([0,T]) for T in T_L])
melt_smooth = f_m*np.array([np.max([0, T]) for T in T_L_smooth])

# plot
plt.plot(time, melt, label="original KAN", alpha=0.5)
plt.plot(time, melt_smooth, label="smooth KAN")
plt.xlim((datetime(2016, 1, 1), datetime(2020, 12, 31)))

L_point = Point(np.mean(df_L.lon), np.mean(df_L.lat))

# L_points = gpd.points_from_xy(x=df_L.lon, y=df_L.lat, crs=4326)
L_gdf = gpd.GeoDataFrame(geometry=[L_point], crs=4326)
L_gdf.to_crs(3413, inplace=True)
x_L = L_gdf.geometry.x[0]
y_L = L_gdf.geometry.y[0]

r = gu.Raster("NETCDF:Greenland_data/MARv3.14-monthly-ERA5_1940_2023.nc:water_input_rate")
delta = r.res[0]*5
r.crop([x_L-delta, y_L-delta, x_L+delta, y_L+delta], inplace=True)
year_0  = 2016 - 1940 # starts in 1940
n_years = 5
b_0 = year_0*12
b_end = b_0 + n_years*12
i_months = range(b_0,b_end+1)
# melt = np.zeros((len(H.dat.data[:]), len(i_months)))  # will interpolate onto same mesh as H
melt_MAR = np.zeros(len(i_months))
for (n,i) in enumerate(i_months):
    melt_MAR[n] = r.interp_points((x_L, y_L), band=i, as_array=True)[0] / 1000 * 12

start_date = datetime(2016, 1, 1)
time_MAR = [start_date + timedelta(days=k*30) for k in range(len(melt_MAR))]
plt.plot(time_MAR, melt_MAR, label="monthly MAR")
plt.legend()

plt.savefig("KAN_smoothing.jpg", dpi=300)
