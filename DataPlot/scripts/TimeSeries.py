import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot import set_figure, unit, getColor, color_maker
from pathlib import Path
from pandas import read_csv, concat, date_range
from DataPlot.data_processing import *
from DataPlot.scripts.Data_classify import state_classify, season_classify, Seasons

PNSD = DataReader('PNSD_dNdlogdp.csv')
PSSD = DataReader('PSSD_dSdlogdp.csv')
PVSD = DataReader('PVSD_dVdlogdp.csv')
PESD = DataReader('PESD_dextdlogdp_internal.csv')

# Time Series
df = DataReader('All_data.csv')
dic_grp_sta = state_classify(df)


def inset_colorbar(ax):

    return ax


def sub(df, trg: str, ax=None, cbar=False, set_visible=False,
        cmap='jet', fig_kws={}, cbar_kws={}, plot_kws={},
        **kwargs):
    cbar_label = cbar_kws.pop('cbar_label', unit(trg))
    cbar_min = cbar_kws.pop('cbar_min', df[trg].min() if df[trg].min() > 0.0 else 1.)
    cbar_max = cbar_kws.pop('cbar_max', df[trg].max())

    # Set the plot_kws
    plot_kws = dict(**plot_kws)

    # Set the figure keywords
    fig_kws = dict(figsize=(10, 4), **fig_kws)

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # main plot
    ax.plot(df.index, df[trg], **plot_kws)

    # Set title
    st_tm, fn_tm = df.index[0], df.index[-1]
    ax.set_title(kwargs.get('title', f'{st_tm.strftime("%Y/%m/%d")} - {fn_tm.strftime("%Y/%m/%d")}'))

    if set_visible:
        ax.axes.xaxis.set_visible(False)

    # Set the figure keywords
    if cbar:
        cbar_kws = dict(label=r'$dN/dlogD_p\ (\# / cm^{-3})$', **cbar_kws)
        clb = plt.colorbar(ax, pad=0.01, **cbar_kws)

    return ax


@set_figure(fs=12)
def time_series(_df):
    st_tm, fn_tm = _df.index[0], _df.index[-1]
    fig, (ax1, ax2, ax3, ax6) = plt.subplots(4, 1, figsize=(12, 6))

    ax1 = sub(_df,
              trg='Extinction',
              ax=ax1,
              plot_kws=dict(lw=1.5, color="b", alpha=1, label=r'$\bf Extinction$'),
              set_visible=True,
              )
    ax1 = sub(_df,
              trg='Scattering',
              ax=ax1,
              plot_kws=dict(lw=1.5, color="g", alpha=1, label=r'$\bf Scattering$'),
              )
    ax1 = sub(_df,
              trg='Absorption',
              ax=ax1,
              plot_kws=dict(lw=1.5, color="r", alpha=1, label=r'$\bf Absorption$'),
              )

    ax1.set_ylabel(r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$')
    ax1.set(ylim=(0., _df.Extinction.max() * 1.1), xlim=(st_tm, fn_tm))
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, labelspacing=0.5, handlelength=1)

    # Temp, RH
    AT, = ax2.plot(time_, _df.AT, color="r", alpha=1)
    ax2.set_ylabel(unit('AT'))
    ax2.set(ylim=(_df.AT.min() - 2, _df.AT.max() + 2), xlim=(st_tm, fn_tm))
    ax2.axes.xaxis.set_visible(False)
    ax2.tick_params(axis='y', colors=AT.get_color())
    ax2.yaxis.label.set_color(AT.get_color())
    ax2.spines['left'].set_color(AT.get_color())

    ax2_2 = ax2.twinx()
    RH, = ax2_2.plot(time_, _df.RH, color="b", alpha=1)
    ax2_2.set_ylabel(unit('RH'))
    ax2_2.set(ylim=(20, 100), xlim=(st_tm, fn_tm))
    ax2_2.axes.xaxis.set_visible(False)

    ax2_2.tick_params(axis='y', colors=RH.get_color())
    ax2_2.yaxis.label.set_color(RH.get_color())
    ax2_2.spines['right'].set_color(RH.get_color())
    ax2_2.spines['left'].set_color(AT.get_color())

    scalar_map, colors = color_maker(_df.PBLH.values)
    ax3.bar(time_, _df.VC, color=scalar_map.to_rgba(colors), width=0.0417, edgecolor='None', linewidth=0)
    # ax3.bar(time_, np.where(colors == 0, df_.VC, 0), width=0, color='None', alpha=1, edgecolor='None', linewidth=0, zorder=2)
    ax3.set_ylabel(r'$\bf VC\ (m^2/s)$')
    ax3.set(ylim=(0, _df.VC.max() * 1.1), xlim=(st_tm, fn_tm))
    ax3.axes.xaxis.set_visible(False)

    ax3_1 = inset_axes(ax3,
                       width="1%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax3.transAxes,
                       borderpad=0,
                       )

    cbar = plt.colorbar(scalar_map, cax=ax3_1, orientation='vertical', ticks=[0, 200, 400, 600, 800])
    cbar.set_label(label=r'$\bf PBLH (m)$')

    ax3_2 = ax3.twinx()
    sc_6 = ax3_2.scatter(time_, _df.WS, c=_df.WD, cmap='hsv', marker='o', s=5, alpha=1.0)
    ax3_2.set_ylabel(r'$\bf WS\ (m/s)$')
    ax3_2.set_ylim((-2, _df.WS.max() * 1.1))
    ax3_2.set_yticks([0, 2, 4])

    ax3_3 = inset_axes(ax3, width="30%", height="5%", loc='upper left')
    color_bar2 = plt.colorbar(sc_6, cax=ax3_3, orientation='horizontal')
    color_bar2.set_label(label=r'$\bf WD $')

    sc_6 = ax6.scatter(time_, _df.PM25, c=_df.PM1 / _df.PM25, vmin=0.2, vmax=1, cmap='jet', marker='o', s=5, alpha=1.0)
    ax6.set_ylabel(r'$\bf PM_{2.5}\ (\mu g/m^3)$')
    ax6.set(ylim=(0, _df.PM25.max() * 1.2), xlim=(st_tm, fn_tm))
    ax4_2 = inset_axes(ax6, width="30%", height="5%", loc='upper left')
    color_bar2 = plt.colorbar(sc_6, cax=ax4_2, orientation='horizontal')
    color_bar2.set_label(label=r'$\bf PM_{1}/PM_{2.5} $')
    # fig.savefig(f'time2_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
    plt.show()


@set_figure(fs=12)
def extinction_timeseries(_df):
    st_tm, fn_tm = _df.index[0], _df.index[-1]
    tick_time = date_range(st_tm, fn_tm, freq='10d')  ## set tick
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    sc = ax1.scatter(_df.index, _df.Extinction, c=_df.PM25, norm=plt.Normalize(vmin=0, vmax=50), cmap='jet',
                     marker='o', s=10, facecolor="b", edgecolor=None, alpha=1)

    ax1.set_title(r'$Extinction\ &\ PM_{2.5}\ Sequence\ Diagram$')
    ax1.set_ylabel('Extinction (1/Mm)')
    ax1.set_ylim(0., )
    ax1.set_xlim(st_tm, fn_tm)

    axins = inset_axes(ax1, width="40%", height="5%", loc=1)
    color_bar = plt.colorbar(sc, cax=axins, orientation='horizontal')
    color_bar.set_label(label=r'$PM_{2.5}$' + r' $({\mu}g/{m^3})$', weight='bold', size=14)
    color_ticks = color_bar.ax.get_xticks().astype(int)
    color_bar.ax.set_xticks(color_ticks)
    color_bar.ax.set_xticklabels(color_ticks, size=14)

    plt.show()


@set_figure(fs=12)
def extinction_month(_df):
    st_tm, fn_tm = _df.index[0], _df.index[-1]
    tick_time = date_range(st_tm, fn_tm, freq='10d')  ## set tick

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(12, 5), dpi=150)

    ax1 = sub(_df, trg='Scattering', ax=ax1)
    sc1 = ax1.scatter(_df.index, _df.Scattering,
                      marker='o', s=15, facecolor="g", edgecolor='k', linewidths=0.3, alpha=0.9)
    sc2 = ax1.scatter(_df.index, _df.Absorption,
                      marker='o', s=15, facecolor="r", edgecolor='k', linewidths=0.3, alpha=0.9)

    ax1.set_ylabel('Scattering & \n Absorption (1/Mm)')
    # ax1.set_xticks(tick_time)
    # ax1.set_xticklabels(tick_time.strftime(''))
    ax1.set_ylim(0, _df['Scattering'].max() + 10)
    ax1.set_xlim(st_tm, fn_tm)
    [ax1.spines[axis].set_visible(False) for axis in ['top']]

    sc3 = ax2.scatter(_df.index, _df.Extinction,
                      marker='o', s=15, facecolor="b", edgecolor='k', linewidths=0.3, alpha=0.9)
    ax2.plot(_df.index, np.full(len(_df.index), _df.Extinction.quantile([0.90])), color='r',
                         ls='--', lw=2, label='therosold')
    ax2.set_ylabel('Extinction (1/Mm)', loc='bottom')
    ax2.set_ylim(0, _df.Scattering.max() + 10)
    ax2.set_xlim(st_tm, fn_tm)

    # ax2.set_title(str(_df.Season[0]) + ' Sequence Diagram')

    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()

    [ax2.spines[axis].set_visible(False) for axis in ['bottom']]
    ax2.get_xaxis().set_visible(False)

    ax2.legend(handles=[sc1, sc2, sc3], labels=['Scattering', 'Absorption', 'Extinction'],
               bbox_to_anchor=(0, 1.), loc='upper left')

    plt.show()


for season, (st_tm_, fn_tm_) in Seasons.items():
    st_tm, fn_tm = pd.Timestamp(st_tm_), pd.Timestamp(fn_tm_)
    IdxTmRange = pd.date_range(st_tm, fn_tm, freq='1h')

    _df = df.loc[st_tm:fn_tm].reindex(IdxTmRange)
    time_ = _df.index

    PNSD_data = PNSD.loc[st_tm:fn_tm].reindex(IdxTmRange)
    PSSD_data = PSSD.loc[st_tm:fn_tm].reindex(IdxTmRange)
    time = PNSD_data.index
    dp = PNSD.keys().astype(float)

    # 數據平滑
    _df = _df.rolling(3).mean(numeric_only=True)

    time_series(_df)
    extinction_timeseries(_df)
    extinction_month(_df)

    break
