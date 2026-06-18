import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from datetime import datetime
from plotting.loading_functions import *
import models_main.helpers as hlp
import pandas as pd
import matplotlib.pyplot as plt

run_indices = np.arange(161,190)


def get_mean_winter_speedup(run_idx):
    timeseries_path = f"parameter_runs/run_{run_idx}/time_series.h5"
    # load model output from hdf5 file
    us_raw, m_raw, phi_raw, q_raw, Q_raw, n_idx = load_model_output(timeseries_path)

    # get model time series
    start_date = datetime(2016, 1, 1)
    dates_model = [start_date + timedelta(days=2*k) for k in range(n_idx)]
    i_model = np.where( (np.array(dates_model) > xstart) &  (np.array(dates_model) < xend) )[0]

    q, Q, Us, pw_pi, m = get_timestamps(mesh_, smesh_, B, H, us_raw, phi_raw, q_raw, Q_raw, m_raw, n_idx)
    Umodel_time = np.zeros((len(Us[0].dat.data_ro),len(i_model))) # space, time
    for n in range(len(i_model)):
        Umodel_time[:,n] = Us[i_model[n]].dat.data_ro
    # calculate monthly averages with panda
    dates = pd.to_datetime(np.array(dates_model)[i_model])
    # create DataFrame with time as index
    df_monthly = pd.DataFrame(
        Umodel_time.T,  # shape (time, space)
        index=dates
    )
    monthly_means = df_monthly.resample('MS').mean()
    # winter_means  = monthly_means.mean(axis=0)

    dU_spatial = monthly_means.diff(axis=0).median(axis=0) #/ winter_means

    # calculate spatial mean
    dU_mean = np.max(dU_spatial)

    return dU_mean


# load run-independent stuff
timeseries_path = f"parameter_runs/run_{run_indices[1]}/time_series.h5"
mesh_, smesh_ = get_meshes(timeseries_path)
B, H, S = load_topography(mesh_)

xstart,xend = datetime(2017,11,1),datetime(2018,4,30)

dUs = [] #np.zeros(len(run_indices))
k_s = [] #np.zeros(len(run_indices))
for (i,r) in enumerate(run_indices):
    print(i)
    try:
        dUs.append(get_mean_winter_speedup(r))
        float_params = hlp.get_params_from_input_file(r)
        k_s.append(float_params["k_s"])
    except:
        continue

plt.scatter(k_s,dUs)
plt.xscale('log')
plt.savefig("test.jpg")
print(dUs)


# ToDo: do this independently for different values of m_basal
# plot the speedup at a certain point only, or in an area, instead of the mean/max of the whole domain?
