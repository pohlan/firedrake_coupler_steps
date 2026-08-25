import firedrake as df
import numpy as np
import models_main.helpers as hlp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import geoutils as gu
import pandas as pd
from scipy.optimize import curve_fit
import glob
from plotting.loading_functions import *
from cycler import cycler

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

def remove_axes(ax):
    ax.set_aspect('equal')
    ax.set_xlim(-2.4e5,-0.2e5)
    ax.set_ylim(-2.585e6,-2.47e6)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

def format_ax(ax, xstart, xend, ig, panel_idx=0, ylims=None, ylabel="", draw_legend=False):
    if not (ylims is None):
        ymin, ymax = ylims
    else:
        ymin, ymax = ax.get_ylim()
    ax.set_xticklabels([])
    ax.vlines([datetime(2018,1,1), datetime(2019,1,1), datetime(2020,1,1), datetime(2021,1,1), datetime(2022,1,1), datetime(2023,1,1), datetime(2024,1,1)], ymin-0.2*(ymax-ymin), ymax+0.2*(ymax-ymin), color="black", ls="dotted", alpha=0.5)
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
        ax1.plot(obs_dates, Uobs-obs_mean, label=f"Observations", color="black", lw=lw, ls="dashed", marker="o", markersize=5)
        # ax1.plot(sorted_dates, Uobs_time, label=f"Observations", color="black", lw=lw, ls="dashed", marker="o", markersize=5)

    # model
    # convert to dataframe in order to calculate 10-day mean
    # dates = pd.to_datetime(dates_model)
    # df_model = pd.DataFrame(Umodel, index=dates)

    # model_resampled = df_model.resample('10D').mean()
    # model_centered = model_resampled.values - model_resampled.values.mean()
    # model_centered = model_resampled.values - model_resampled.values[model_resampled.values < 300].mean()
    # model_centered[model_centered > 150] = np.nan
    # ax1.plot(model_resampled.index, model_centered, label=model_label, color=color, ls="solid", lw=lw)
    model_mean = np.mean(np.array(Umodel)[np.where(np.isfinite(Umodel))])
    ax1.plot(dates_model, Umodel-model_mean, label=model_label, color=color, ls="solid", lw=lw)

    # ymin = min(np.array(Umod_time)[i_model].min()-model_mean, np.min(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])-obs_mean)
    # ymax = max(np.array(Umod_time)[i_model].max()-model_mean, np.max(np.array(Uobs_time)[i_obs[np.where(np.isfinite(np.array(Uobs_time)[i_obs]))[0]]])-obs_mean)
    format_ax(ax1, xstart, xend, ig, panel_idx=ig, ylims=(-80,140), ylabel=r"Speed rel. to mean ($\mathrm{m^3\,a^{-1}}$)")
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
        ax2.set_ylim(0, 19)
        ax2.tick_params(axis='y', labelright=False, colors="grey", length=7, width=2)
        if (ig==n_glaciers-1):
            ax2.set_ylabel(r"Runoff ($\mathrm{m^3\,a^{-1}}$)")
            ax2.yaxis.label.set_color("grey")
            ax2.tick_params(axis='y', labelright=True, colors="grey", length=7, width=2)

    return

def fit_linear_slope(U, sig=None):
    # print("do curve fit...")
    def linear(t, intercept, slope):
            return intercept + slope * t
    t = range(len(U))
    if sig is None:
        popt, pcov = curve_fit(linear, t, U)
        intercept, slope = popt
        return slope
    else:
        popt, pcov = curve_fit(linear, t, U, sigma=sig, absolute_sigma=True)
        intercept, slope = popt
        slope_err = np.sqrt(pcov[1, 1])
        return slope, slope_err

def panel_letter_annotation(ax, ii, annotate_fs, xyc=(0.03, 1.0)):
    ax.annotate(
                f"{idx_to_letter(ii)}",
                xy=xyc,
                xycoords="axes fraction",
                fontsize=annotate_fs * 0.9,
                fontweight="bold",
            )

# Fig. 4 and Fig. SXX

def plot_sensitivity_panel(ax, mesh_, vel_dir, run_indices, var1, var2, xstart, xend, xc, yc, colors, ii, param_units, param_labels,
                           xscale="log", labelloc="best", legend=True, lw_times=2, marker_size_times=8, annotate_fs=23, ylims=None):
    dUs = []
    param1 = []
    param2 = []

    for run_idx in run_indices:
        try:
            U_monthly = get_monthly_means_model(mesh_, run_idx, xstart, xend, xc, yc, start_year=2014)
            if len(U_monthly) == 0:
                continue
            dU = fit_linear_slope(U_monthly)
            dUs.append(dU)
            float_params = hlp.get_params_from_input_file(run_idx)
            param1.append(float_params[var1])
            param2.append(float_params[var2])
            print(f"run {run_idx}, {var1} = {float_params[var1]}")
        except:
            continue

    param1 = np.array(param1)
    dUs = np.array(dUs)

    obs_files = glob.glob(vel_dir+"*vv*.tif")
    obs_dates, obs_files = get_obs_files(obs_files, xrange=(xstart,xend))
    Uobs_glaciers = load_obs_timeseries(obs_files, xc, yc)[:, 0]
    # obs_m = np.mean(np.diff(Uobs_glaciers))
    sig = get_obs_errors(xstart, xend, xc, yc, vel_dir)
    slope, slope_err = fit_linear_slope(Uobs_glaciers, sig)

    ax.set_prop_cycle(cycler(color=colors))
    for m in np.unique(param2):
        i_m = np.where(param2 == m)
        idx = np.argsort(param1[i_m])
        ax.plot(param1[i_m][idx], dUs[i_m][idx], label=f"{m}", marker="o", lw=lw_times, markersize=marker_size_times)

    xmin, xmax = ax.get_xlim()
    ax.fill_between([xmin,xmax], slope-2*slope_err, slope+2*slope_err, alpha=0.1, color="black")
    ax.hlines(slope, xmin, xmax, color="black", ls="dashed", lw=lw_times)
    # ax.legend(title=r"$m_\mathrm{basal}$ ($\mathrm{m\,a^{-1}}$)")
    if param_units[var1] == "":
        l1 = param_labels[var1]
    else:
        l1 = param_labels[var1]+" ("+ param_units[var1] +")"
    if param_units[var2] == "":
        l2 = param_labels[var2]
    else:
        l2 = param_labels[var2]+" ("+ param_units[var2] +")"
    if legend:
        ax.legend(title=l2, loc=labelloc)
    ax.set_xlabel(l1)
    ax.set_ylabel(r"$\Delta u$ ($\mathrm{m\,a^{-1}}$)")
    ax.set_xscale(xscale)
    panel_letter_annotation(ax, ii, annotate_fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylims is not None:
        ax.set_ylim(ylims)
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    if xscale == "log":
        xannotate = np.exp(np.log(x0) + 0.46*(np.log(x1)-np.log(x0)))
    else:
        xannotate = x0+0.46*(x1-x0)
    ax.annotate("Observed", xy=(xannotate, slope+2*slope_err+0.02*(y1-y0)), fontsize=annotate_fs * 0.9, annotation_clip=False)


def plot_timeseries_panel(ax, mesh_, vel_dir, run_indices, var, xstart, xend, xstart_plot, xend_plot, xc, yc, cols, ii, param_units, param_labels,
                          lw_times=2, marker_size_times=8, annotate_fs=23, ylims=None):
    files = glob.glob(vel_dir+"*vv*.tif")
    obs_dates, obs_files = get_obs_files(files, xrange=(xstart_plot, xend_plot))
    for (run_idx,c) in zip(run_indices,cols):
        float_params = hlp.get_params_from_input_file(run_idx)
        param = float_params[var]
        # m  = float_params["m_basal"]
        dates_model, U_model = get_model_timeseries_for_locations(
            mesh_, run_idx, xstart_plot, xend_plot, xc, yc, start_year=2014
        )
        lab = param_labels[var] + f" = {param} " + param_units[var]
        ax.plot(dates_model, U_model - np.mean(U_model), color=c, label=lab, lw=lw_times) #+"\n"+rf"$m_\mathrm{{basal}}= {m}$")

    Uobs_glaciers = load_obs_timeseries(obs_files, xc, yc)[:, 0]
    ax.plot(obs_dates, Uobs_glaciers - np.nanmean(Uobs_glaciers), color="black", ls="dashed", marker="o", label="Observed", lw=lw_times, markersize=marker_size_times)
    if ylims is not None:
        y0, y1 = ylims
    else:
        y0, y1 = ax.get_ylim()
    ax.fill_betweenx([y0, y1], xstart, xend, alpha=0.15, facecolor="cornflowerblue")
    ax.set_xlabel("Month of 2020/2021")
    ax.set_ylabel(r"Speed rel. to mean ($\mathrm{m\,a^{-1}}$)")
    ax.legend()
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    panel_letter_annotation(ax, ii, annotate_fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
