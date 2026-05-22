import firedrake as df
import numpy as np
import models_main.helpers as hlp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def slice(s,s0,s1):
    return df.conditional(df.And(s > s0, s<=s1), 1.0, 0.0)

def get_flux_ratio(q, Q, smesh_, s, s_sub, s0s):
    Qi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi = slice(s, s0, s1)
        chi_s = slice(s_sub, s0, s1)
        # Integrate q over bulk domain and Q over submesh (boundary)
        Q_flux = df.assemble(df.avg(Q)*chi_s*df.dx(domain=smesh_)) / df.assemble(chi_s*df.dx(domain=smesh_)) # on the submesh, dx is along traces, so 1D not 2D
        q_flux = df.assemble(q*chi*df.dx) / df.assemble(chi*df.dx)
        Qi.append(Q_flux/(Q_flux+q_flux))
    return Qi

def get_Q(Q, smesh_, s_sub, s0s):
    Qi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi_s = slice(s_sub, s0, s1)
        Q_avg = df.assemble(Q*chi_s*df.dx(domain=smesh_)) / df.assemble(chi_s*df.dx(domain=smesh_)) # average
        Qi.append(Q_avg)
    return Qi

def get_variable(X, mesh_, s, s0s, mask=None):
    # get function space
    V_DG0 = df.FunctionSpace(mesh_, "DG", 0)
    # loop through points along flowline
    Xi = []
    for (s0,s1) in zip(s0s[:-1],s0s[1:]):
        chi = slice(s, s0, s1)
        chi2 = df.Function(V_DG0)
        if mask is None:
            chi2.interpolate(1.0)
        elif df.assemble(chi*mask*df.dx) / df.assemble(chi*df.dx) < 0.1:
            Xi.append(np.nan)
            continue
        else:
            chi2.interpolate(mask)
        # print(len(np.where(np.isfinite(X.dat.data_ro))))
        # chi2.dat.data[np.where(np.isfinite(X.dat.data_ro))] = 1.0
        X_avg = df.assemble(X*chi*chi2*df.dx) / df.assemble(chi*chi2*df.dx) # average
        Xi.append(X_avg)
    return Xi

def idx_to_month(idx, dt=2):
    # dt is timestamp of model output in years
    dec = np.round((idx*dt/365)%1,decimals=2)
    if dec == 0:
        return "Jan"
    elif dec == 0.25:
        return "April"
    elif dec == 0.5:
        return "July"
    elif dec == 0.7:
        return "Sep"

def format_ax(ax, xstart, xend, ig, ylims=None, ylabel="", draw_legend=True):
    if not (ylims is None):
        ymin, ymax = ylims
    else:
        ymin, ymax = ax.get_ylim()
    ax.set_xticklabels([])
    ax.vlines([datetime(2019,1,1), datetime(2020,1,1), datetime(2021,1,1), datetime(2022,1,1), datetime(2023,1,1)], ymin-0.2*(ymax-ymin), ymax+0.2*(ymax-ymin), color="black", ls="dotted", alpha=0.5)
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1,7]))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(xstart,xend)
    # ax.set_ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel(ylabel)
    if draw_legend:
        plt.legend()
    if ig > 0:
        ax.tick_params(axis='y', labelleft=False, length=7, width=2)

def plot_vel_timeseries(mesh_, splus, s, dates_model, Us, m, sorted_dates, U_obs, U_mask, xstart, xend, i_model, color, lw, ds, ig, n_glaciers):
    # get function space coordinates
    meshx_DG0, meshy_DG0 = zip(*hlp.get_coordinates(mesh_, "DG", 0))
    meshx, meshy = zip(*hlp.get_coordinates(mesh_, "CG", 1))
    # get coordinates of points to plot
    p_DG0  = np.argmin(abs(s.dat.data_ro-splus))
    xi, yi = meshx_DG0[p_DG0], meshy_DG0[p_DG0]
    p_CG1  = np.argmin(np.sqrt((xi-meshx)**2+(yi-meshy)**2))
    xi2, yi2 = meshx[p_CG1], meshy[p_CG1]

    Umod_time = []
    Uobs_time = []
    m_time    = []
    # for i in range(n_idx):
        # with CheckpointFile(timeseries_path, 'r') as afile:
            # Umod_time.append(afile.load_function(mesh_, "Us", idx=i).dat.data_ro[p_CG1])
    for (Umod,m_) in zip(Us,m):
        Umod_time.append(get_variable(Umod, mesh_, s, [splus-ds/2,splus+ds/2])[0])
        m_time.append(get_variable(m_, mesh_, s, [splus-ds/2,splus+ds/2])[0])
    for (Uobs,mask) in zip(U_obs, U_mask):
        # Uobs_time.append(Uobs.dat.data_ro[p_DG0])
        Uobs_time.append(get_variable(Uobs, mesh_, s, [splus-ds/2,splus+ds/2],mask=mask)[0])
    model_mean = np.array(Umod_time)[i_model].mean()
    plt.plot(dates_model, Umod_time-model_mean, label=f"model", color=color, ls="solid", lw=lw)
    i_obs = np.where( (np.array(sorted_dates) > xstart) &  (np.array(sorted_dates) < xend) )[0]
    obs_mean = np.mean(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])
    plt.plot(sorted_dates, Uobs_time-obs_mean, label=f"observations", color="black", lw=lw, ls="dashed")
    ymin = min(np.array(Umod_time)[i_model].min()-model_mean, np.min(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])-obs_mean)
    ymax = max(np.array(Umod_time)[i_model].max()-model_mean, np.max(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])-obs_mean)
    ax = plt.gca()
    format_ax(ax, xstart, xend, ig, ylims=(-80,105), ylabel="Speed rel. to mean (m/yr)")
    # melt input
    ax2 = ax.twinx()
    ax2.fill_between(dates_model, m_time, label="melt", color="grey", alpha=0.3)
    ax2.tick_params(axis='y', labelright=False, colors="grey", length=7, width=2)
    if (ig==n_glaciers-1):
        ax2.set_ylabel("Runoff (m/yr)")
        ax2.yaxis.label.set_color("grey")
        ax2.tick_params(axis='y', labelright=True, colors="grey", length=7, width=2)
    return ax2
