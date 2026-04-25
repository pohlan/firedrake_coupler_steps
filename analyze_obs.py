import os
import firedrake as df
from firedrake.__future__ import interpolate
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor, triplot
import xarray as xr
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit
from datetime import datetime
import re


# load observations and interpolate onto mesh
mesh = df.Mesh("Greenland_data/russel/russel.msh")
v_dg = df.VectorFunctionSpace(mesh, "CG", 1)
X = df.assemble(interpolate(mesh.coordinates,v_dg))
meshx = X.dat.data_ro[:,0]
meshy = X.dat.data_ro[:,1]
vel_dir = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
files   = os.listdir(vel_dir)

# extract datetime object from filenames and sort after date
date_pattern = r'\d{2}[A-Z][a-z]{2}\d{2}'
format_string = "%d%b%y"
date_list = []
for f in files:
    date_str    = re.search(date_pattern, f).group(0)
    date_object = datetime.strptime(date_str, format_string)
    date_list.append(date_object)
pairs = zip(date_list, files)
sorted_dates, sorted_files = zip(*sorted(pairs))

# read files in order
n_months = len(files)
Us_obs = np.zeros((len(meshx),n_months))
for (i,f) in enumerate(sorted_files[:n_months]):
    r = gu.Raster(f"{vel_dir}{f}")
    delta = r.res[0]*2
    r.crop([min(meshx)-delta, min(meshy)-delta, max(meshx)+delta, max(meshy)+delta], inplace=True)
    Us_obs[:,i] = r.interp_points((meshx,meshy))


coords = [[-223885, -2.49326e6],
          [-217000, -2.49416e6],
          [-211000, -2.49506e6],
          [-205468, -2.49596e6],
          [-225130, -2.50269e6],
          [-216304, -2.50337e6],
          [-218312.472667319,-2514023.04736881],
          [-215217.697420904,-2510443.87251861],
          [-203042.0,-2507072.0],
          [-192320.0,-2508747.0]
          ]


#############################
# plot average seasonal signal for different points

plt.figure(figsize=(6,6))
for (i,(xi,yi)) in enumerate(coords):
    p = np.argmin(np.sqrt((meshx-xi)**2+(meshy-yi)**2))
    data = {'date': sorted_dates[:], 'value': Us_obs[p,:]}
    dfr = pd.DataFrame(data)
    dfr.set_index('date', inplace=True)

    mean_seasonal_cycle = dfr.groupby(dfr.index.month).mean()
    # ax = plt.subplot(4,2,i+1)
    speed = np.concatenate([mean_seasonal_cycle[4:],mean_seasonal_cycle[:4]])
    plt.plot(speed / np.mean(speed), label=f"{i}")
    # ax.plot(speed / np.mean(speed), label=f"{i}")
    # plt.title(f"{i}", pad=2.0)
    plt.ylabel("Surface speed (m/yr)")
    plt.legend()
    plt.savefig("test.jpg")

# plot map
x0, xend = -2.35e5,-1.75e5
y0, yend = -2.55e6, -2.488e6
backg = gu.Raster(f"{vel_dir}{files[0]}")
delta = backg.res[0]*2
backg.crop([x0, y0, xend, yend], inplace=True)
plt.figure()
backg.plot()
for(i,(x_crd,y_crd)) in enumerate(coords):
    plt.scatter([x_crd],[y_crd], color="black")
    plt.annotate(text=f"{i}", xy=(x_crd+5e2,y_crd+5e2), fontsize=14)
plt.savefig(f"parameter_runs/plots/map_analyze.jpg")


####################################
# define asymmetric gaussian
####################################


def gaussian_asym(x,x0,A,sigma2_left, sigma2_right, m, q, y0):
    # x0=4,A=10,sigma2_left=4, sigma2_right=1, m=0.5, q=-4
    y_lin = m*x + q
    i_left = np.where(x < x0)
    sigma2 = np.ones(len(x)) * sigma2_right
    sigma2[i_left] = sigma2_left
    # if x < x0:
    #     sigma2 = sigma2_left
    # else:
    #     sigma2 = sigma2_right
    return np.maximum(A*np.exp(-(x-x0)**2/(2*sigma2)), y_lin) + y0

def gaussian(x, x0, A, sigma2, m, q, y0):
    y_lin = m*x + q
    return np.maximum(A*np.exp(-(x-x0)**2/(2*sigma2)), y_lin) + y0

xs = np.arange(0,12,0.1)
# ys = gaussian_asym(xs, 4, 10, 4, 1, 0.5, -4)
ys = gaussian(xs, 1.8, 0.6, 1.7, 0.005, 0.0, 0.8)

plt.figure()
plt.plot(xs, ys)
plt.savefig("test_gaussian.jpg")

####################################
# take one particular point and fit
#######################################

xi, yi = coords[1]
p = np.argmin(np.sqrt((meshx-xi)**2+(meshy-yi)**2))
# p = 2162
data = {'date': sorted_dates[:], 'value': Us_obs[p,:]}
dfr = pd.DataFrame(data)
dfr.set_index('date', inplace=True)

mean_cycle = dfr.groupby(dfr.index.month).mean()
# ax = plt.subplot(4,2,i+1)
speed = np.concatenate([mean_cycle[4:],mean_cycle[:4]])[:,0]
speed = speed / np.nanmean(speed)

popt, pcov = curve_fit(gaussian, np.linspace(0,12,12), speed,
                       p0=[1.8, 0.6, 1.7, 0.005, 0.0, 0.8],
                    #    bounds=([1,0.4,1,0,-0.1,0.5],[3,1,2.5,0.01,0.2,1])
                       )


# x0, A, sigma2, m, q, y0 = popt
# x0,A,sigma2_left, sigma2_right, m, q, y0 = popt
# y_rec = gaussian_asym(xs, x0,A,sigma2_left, sigma2_right, m, q, y0)
# y_rec = gaussian_asym(xs, 4.0, 0.6, 2, 0.5, 0.2, 0.2, 0.7)

x0, A, sigma2, m, q, y0 = popt
y_rec = gaussian(xs, x0, A, sigma2, m, q, y0)
plt.figure()
# plt.plot(sorted_dates, Us_obs[p,:]/np.nanmean(Us_obs[p,:]), label="obs")
plt.plot(speed,label="obs")
plt.plot(xs, y_rec)
plt.legend()
plt.savefig("test_gaussian.jpg")


####################################
# loop through all coords in domain
#######################################

dfr_all = pd.DataFrame(np.transpose(Us_obs))
dfr_all.insert(loc=0, column='date', value=sorted_dates[:])
dfr_all.set_index('date', inplace=True)
mean_seasonal_cycle = dfr_all.groupby(dfr_all.index.month).mean()

x0s = np.zeros(Us_obs.shape[0])
As  = np.zeros(Us_obs.shape[0])
sigma2s = np.zeros(Us_obs.shape[0])
sigma2s_right = np.zeros(Us_obs.shape[0])
ms = np.zeros(Us_obs.shape[0])
qs = np.zeros(Us_obs.shape[0])
y0s = np.zeros(Us_obs.shape[0])
y_recs = np.zeros((Us_obs.shape[0], len(xs)))
for (i,(column_name, mean_season)) in enumerate(mean_seasonal_cycle.items()):
    if column_name == 'date':
        continue
    speed = np.concatenate([mean_season[3:],mean_season[:3]])
    speed = speed / np.nanmean(speed)
    i_nonan = np.where(~np.isnan(speed))
    if len(i_nonan[0]) < 7:
        print(f"Too many nans at point {i}.")
        continue
    try:
        popt, pcov = curve_fit(gaussian, np.linspace(0,12,12)[i_nonan], speed[i_nonan],
                       p0=[1.8, 0.6, 1.7, 0.005, 0.0, 0.8],
                       bounds=([1, 0.4, 0.5, 0, -0.1, 0.5],[4, 2, 7, 0.1, 0.4, 1])
                       )
        x0, A, sigma2, m, q, y0 = popt
        y_rec = gaussian(xs, x0, A, sigma2, m, q, y0)

        # popt, pcov = curve_fit(gaussian_asym, np.linspace(0,12,12)[i_nonan], speed[i_nonan],
        #                p0=[1.8, 0.6, 1.7, 1.7, 0.005, 0.0, 0.8],
        #                bounds=([1, 0.4, 0.5, 0.5, 0, -10, 0.5],[6, 2, 10, 10, 0.5, 0.4, 1])
        #                )
        # x0,A,sigma2_left, sigma2_right, m, q, y0 = popt
        # y_rec = gaussian_asym(xs,x0,A,sigma2_left, sigma2_right, m, q, y0)
    except RuntimeError:
        print(f"No solution achieved for point {i}.")
        x0, A, sigma2, m, q, y0 = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        # x0, A, sigma2_left, sigma2_right, m, q, y0 = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        y_rec = np.nan

    # save parameters and y_rec in a matrix or dataframe
    x0s[i] = x0
    As[i] = A
    sigma2s[i] = sigma2
    # sigma2s_right[i] = sigma2_right
    ms[i] = m
    qs[i] = q
    y0s[i] = y0
    y_recs[i,:] = y_rec


x0, xend = -2.35e5,-1.75e5
y0, yend = -2.55e6, -2.488e6

V = df.FunctionSpace(mesh, "CG", 1)
X0 = df.Function(V)
X0.vector()[:] = sigma2s
fig, axes = plt.subplots()
cl = tripcolor(X0, axes=axes)
plt.xlim(x0,xend)
plt.ylim(y0,yend)
# cl.set_clim(0.4, 1.2)
fig.colorbar(cl)
plt.savefig("parameter_runs/plots/sigma2s.jpg", dpi=300)



coords = [[-223885, -2.49326e6],
          [-205468, -2.49596e6],
          [-225130, -2.50269e6],
          [-216304, -2.50337e6],
          [-218312.472667319,-2514023.04736881],
          [-215217.697420904,-2510443.87251861],
          [-203042.0,-2507072.0],
          [-192320.0,-2508747.0]]
xi, yi = coords[4]
p = np.argmin(np.sqrt((meshx-xi)**2+(meshy-yi)**2))
mean_season = mean_seasonal_cycle.iloc[:,p+1]
speed = np.concatenate([mean_season[3:],mean_season[:3]])
speed = speed / np.nanmean(speed)
plt.figure()
plt.plot(speed, label="obs")
plt.plot(xs, y_recs[p,:])
plt.legend()
plt.savefig("test_gaussian.jpg")

####################################







########################################

# def season_f(x,A,cutoff=0):
#     return np.maximum(A*np.sin(x*2*np.pi/12), cutoff)

# months = np.array([m.month for m in sorted_dates])
# y_f    = np.array([season_f(m,2,cutoff=-0.5) for m in months])

# plt.figure(figsize=(12,3))
# # plt.plot(x, np.sin(x*(2*np.pi/20)))
# plt.plot(sorted_dates[0:30], y_f[0:30])
# plt.savefig("test.jpg")






# plt.figure(figsize=(12,14))
# for (i,(xi,yi)) in enumerate(coords):
#     p = np.argmin(np.sqrt((meshx-xi)**2+(meshy-yi)**2))
#     ax = plt.subplot(4,2,i+1)
#     plt.plot(dates_model[:-1], Us_model[p,1:] / np.mean(Us_model[p,1:]), label="model")
#     plt.plot(sorted_dates[:n_months], Us_obs[p,:]/np.nanmean(Us_obs[p,:]), label="obs")
#     plt.title(f"{i}", pad=2.0)
#     plt.ylabel("Surface speed (m/yr)")
#     plt.legend()
# plt.savefig(f"parameter_runs/plots/run_{run_index}.jpg")



