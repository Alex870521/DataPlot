import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot import set_figure, unit, getColor, color_maker
from pandas import DataFrame, concat, date_range
from DataPlot.data_processing import *
from DataPlot.scripts.Data_classify import state_classify, season_classify, Seasons


def inset_colorbar(ax, axes_image, label=None, loc=None, **kwargs):
    axis = inset_axes(ax,
                      width="35%",
                      height="7%",
                      loc=loc or 'lower left',
                      bbox_to_anchor=(0.015, 0.85, 1, 1),
                      bbox_transform=ax.transAxes,
                      borderpad=0,
                      )
    color_bar = plt.colorbar(axes_image, cax=axis, orientation='horizontal')
    color_bar.set_label(label=label or '', weight='bold', size=12)

    # set color_ticks
    # color_ticks = kwargs.pop('ticks', color_bar.ax.get_xticks()).astype(int)
    # color_bar.ax.set_xticks(color_ticks)
    # color_bar.ax.set_xticklabels(color_ticks, size=12)
    return axis


def sub_timeseries(df, target_col: str, c: str = None, *, ax=None, cbar=False, set_visible=False,
                   fig_kws={}, cbar_kws={}, plot_kws={}, **kwargs):

    # Set cbar_kws
    cbar_label = cbar_kws.pop('cbar_label', unit(target_col))
    cbar_min = cbar_kws.pop('cbar_min', df[target_col].min() if df[target_col].min() > 0.0 else 1.)
    cbar_max = cbar_kws.pop('cbar_max', df[target_col].max())
    cmap = cbar_kws.pop('cmap', 'jet')

    # Set the plot_kws
    plot_kws = dict(marker='o', s=10, edgecolor=None, linewidths=0.3, alpha=0.9, **plot_kws)

    # Set the figure keywords
    fig_kws = dict(**fig_kws)

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # main plot
    # ax.plot(df.index, df[trg], **plot_kws)
    ax.scatter(df.index, df[target_col], **plot_kws)

    # set title
    st_tm, fn_tm = df.index[0], df.index[-1]
    ax.set_title(kwargs.get('title', f'{st_tm.strftime("%Y/%m/%d")} - {fn_tm.strftime("%Y/%m/%d")}'))

    # set ticks
    freq = kwargs.get('freq', '10d')
    tick_time = date_range(st_tm, fn_tm, freq=freq)

    ax.set_xticks(tick_time)
    ax.set_xticklabels(tick_time, size=12)

    if set_visible:
        ax.axes.xaxis.set_visible(False)

    # Set the figure keywords
    if cbar:
        cbar_kws = dict(label=r'$dN/dlogD_p\ (\# / cm^{-3})$', **cbar_kws)
        clb = plt.colorbar(ax, pad=0.01, **cbar_kws)

    return ax


@set_figure(fs=12)
def time_series(df):
    st_tm, fn_tm = df.index[0], df.index[-1]
    tick_time = date_range(st_tm, fn_tm, freq='10d')  ## set tick

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(len(df.index) * 0.02, 6))

    ax1 = sub_timeseries(df,
                         target_col='Extinction',
                         ax=ax1,
                         plot_kws=dict(color="b", label=r'$\bf Extinction$'),
                         set_visible=True,
                         )
    ax1 = sub_timeseries(df,
                         target_col='Scattering',
                         ax=ax1,
                         plot_kws=dict(color="g", label=r'$\bf Scattering$'),
                         )
    ax1 = sub_timeseries(df,
                         target_col='Absorption',
                         ax=ax1,
                         plot_kws=dict(color="r", label=r'$\bf Absorption$'),
                         )

    ax1.set_ylabel(r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$')
    ax1.set(ylim=(0., df.Extinction.max() * 1.1), xlim=(st_tm, fn_tm))
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, labelspacing=0.5, handlelength=1)

    # Temp, RH
    AT, = ax2.plot(time_, df.AT, color="r", alpha=1)
    ax2.set_ylabel(unit('AT'))
    ax2.set(ylim=(df.AT.min() - 2, df.AT.max() + 2), xlim=(st_tm, fn_tm))
    ax2.axes.xaxis.set_visible(False)
    ax2.tick_params(axis='y', colors=AT.get_color())
    ax2.yaxis.label.set_color(AT.get_color())
    ax2.spines['left'].set_color(AT.get_color())

    ax2_2 = ax2.twinx()
    RH, = ax2_2.plot(time_, df.RH, color="b", alpha=1)
    ax2_2.set_ylabel(unit('RH'))
    ax2_2.set(ylim=(20, 100), xlim=(st_tm, fn_tm))
    ax2_2.axes.xaxis.set_visible(False)

    ax2_2.tick_params(axis='y', colors=RH.get_color())
    ax2_2.yaxis.label.set_color(RH.get_color())
    ax2_2.spines['right'].set_color(RH.get_color())
    ax2_2.spines['left'].set_color(AT.get_color())

    scalar_map, colors = color_maker(df.PBLH.values)
    ax3.bar(time_, df.VC, color=scalar_map.to_rgba(colors), width=0.0417, edgecolor='None', linewidth=0)
    # ax3.bar(time_, np.where(colors == 0, df_.VC, 0), width=0, color='None', alpha=1, edgecolor='None', linewidth=0, zorder=2)
    ax3.set_ylabel(r'$\bf VC\ (m^2/s)$')
    ax3.set(ylim=(0, df.VC.max() * 1.1), xlim=(st_tm, fn_tm))
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
    sc_6 = ax3_2.scatter(time_, df.WS, c=df.WD, cmap='hsv', marker='o', s=5, alpha=1.0)
    ax3_2.set_ylabel(r'$\bf WS\ (m/s)$')
    ax3_2.set_ylim((-2, df.WS.max() * 1.1))
    ax3_2.set_yticks([0, 2, 4])

    # add colorbar
    ax3_3 = inset_colorbar(ax3, sc_6, label=r'$\bf WD $')

    sc_6 = sub_timeseries(df,
                          target_col='PM25',
                          ax=ax4,

                          plot_kws=dict(color="g", label=r'$\bf PM_{2.5}\ (\mu g/m^3)$'),
                          )

    sc_6 = ax4.scatter(time_, df.PM25, c=df.PM1 / df.PM25, vmin=0.2, vmax=1, cmap='jet', marker='o', s=5, alpha=1.0)

    ax4.set(ylim=(0, df.PM25.max() * 1.2), xlim=(st_tm, fn_tm))

    # add colorbar
    ax4_2 = inset_colorbar(ax4, sc_6, label=r'$\bf PM_{1}/PM_{2.5} $')

    # fig.savefig(f'time2_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
    plt.show()


if __name__ == '__main__':
    PNSD = DataReader('PNSD_dNdlogdp.csv')
    PSSD = DataReader('PSSD_dSdlogdp.csv')
    PVSD = DataReader('PVSD_dVdlogdp.csv')
    PESD = DataReader('PESD_dextdlogdp_internal.csv')
    df = DataReader('All_data.csv')

    # Season timeseries
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

        break
