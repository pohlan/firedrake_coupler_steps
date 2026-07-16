import firedrake as df
import numpy as np
import models_main.helpers as hlp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import geoutils as gu
import pandas as pd

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

def idx_to_letter(i):
    return chr(ord('a') + i)

def format_ax(ax, xstart, xend, ig, panel_idx=0, ylims=None, ylabel="", draw_legend=False):
    if not (ylims is None):
        ymin, ymax = ylims
    else:
        ymin, ymax = ax.get_ylim()
    ax.set_xticklabels([])
    ax.vlines([datetime(2018,1,1), datetime(2019,1,1), datetime(2020,1,1), datetime(2021,1,1)], ymin-0.2*(ymax-ymin), ymax+0.2*(ymax-ymin), color="black", ls="dotted", alpha=0.5)
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1]))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', length=0, width=2)  # no ticks
    # ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(xstart,xend)
    # ax.set_ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel(ylabel)
    if draw_legend:
        ax.legend()
    if ig > 0:
        ax.tick_params(axis='y', labelleft=False, length=7, width=2)
    else:
        ax.tick_params(axis='y', length=7, width=2)
    # panel annotation
    panel_label = idx_to_letter(panel_idx)
    ax.annotate(panel_label, xy=(0.03,0.9), xycoords="axes fraction", fontsize=18, fontweight="bold")

def plot_vel_timeseries(dates_model, Umodel, m_time, obs_dates, Uobs, xstart, xend, color, lw, ig, n_glaciers, model_label, ax1):
    # observations, plot only if the ax is empty still to avoid re-doing it several times
    if len(ax1.lines) == 0:
        obs_mean = np.mean(np.array(Uobs)[np.where(np.isfinite(Uobs))])
        ax1.plot(obs_dates, Uobs-obs_mean, label=f"observations", color="black", lw=lw, ls="dashed", marker="o", markersize=5)
        # ax1.plot(sorted_dates, Uobs_time, label=f"Observations", color="black", lw=lw, ls="dashed", marker="o", markersize=5)

    # model
    # convert to dataframe in order to calculate 10-day mean
    dates = pd.to_datetime(dates_model)
    df_model = pd.DataFrame(Umodel, index=dates)

    model_resampled = df_model.resample('10D').mean()
    model_centered = model_resampled.values - model_resampled.values.mean()
    ax1.plot(model_resampled.index, model_centered, label=model_label, color=color, ls="solid", lw=lw)
    # ax1.plot(dates_model[1:], Umod_time[1:], label=model_label, color=color, ls="solid", lw=lw)

    # ymin = min(np.array(Umod_time)[i_model].min()-model_mean, np.min(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])-obs_mean)
    # ymax = max(np.array(Umod_time)[i_model].max()-model_mean, np.max(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])-obs_mean)
    format_ax(ax1, xstart, xend, ig, ylims=(-80,150), ylabel=r"Speed rel. to mean ($\mathrm{m^3\,a^{-1}}$)")
    # format_ax(ax1, xstart, xend, ig, ylims=(50,250), panel_idx=ig, ylabel=r"Speed ($\mathrm{m^3\,a^{-1}}$)")

    # melt input, also only plot once the first time
    if len(ax1.lines) == 2:
        ax2_0 = ax1.inset_axes([0, 0, 1, 0.3], transform=ax1.transAxes, sharex=ax1)
        ax2_0.patch.set_visible(False)          # no background
        ax2_0.tick_params(axis='y', labelleft=False, length=0)
        ax2_0.tick_params(axis='x', labelbottom=False, length=0)
        for spine in ax2_0.spines.values():
            spine.set_visible(False)            # no box
        ax2 = ax2_0.twinx()
        ax2.patch.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)            # no box
        ax2.fill_between(dates_model, m_time, color="grey", alpha=0.3)
        ax2.tick_params(axis='y', labelright=False, colors="grey", length=7, width=2)
        if (ig==n_glaciers-1):
            ax2.set_ylabel(r"Runoff ($\mathrm{m^3\,a^{-1}}$)")
            ax2.yaxis.label.set_color("grey")
            ax2.tick_params(axis='y', labelright=True, colors="grey", length=7, width=2)

    return
