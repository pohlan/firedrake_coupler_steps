import os
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import firedrake as df
from firedrake.__future__ import interpolate
from firedrake.output import VTKFile
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
T_L_smooth = lowess(T_L, time, frac=1/200, xvals=time)

data_dir = 'Greenland_data/'
dff = pd.DataFrame({'dT_days':dT_days, 'T_L': T_L, 'T_L_smooth':T_L_smooth, 'z_L': z_L})
dff.to_csv(data_dir+'KAN_melt.csv',index=False)
