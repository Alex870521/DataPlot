import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import DataFrame, concat, date_range
from DataPlot.data_processing import *
from DataPlot.plot import *


def inset_colorbar(ax, axes_image, orientation, inset_kws={}, clb_kws={}):
    # Set inset_axes_kws
    if orientation == 'horizontal':
        inset_kws.update(width="35%",
                         height="7%",
                         loc='lower left',
                         bbox_to_anchor=(0.015, 0.85, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0,
                         )

    if orientation == 'vertical':
        inset_kws.update(width="1%",
                         height="100%",
                         loc='lower left',
                         bbox_to_anchor=(1.1, 0., 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0,
                         )

    # Set clb_kws
    clb_kws = dict(**clb_kws)

    axis = inset_axes(ax, **inset_kws)
    color_bar = plt.colorbar(axes_image, cax=axis, orientation=orientation, **clb_kws)

    return axis


def sub_timeseries(df, target_col, c=None, ax=None, cbar=False, set_visible=True,
                   fig_kws={}, cbar_kws={}, plot_kws={}, **kwargs):
    time = df.index.copy()

    # Set cbar_kws
    cbar_label = cbar_kws.pop('cbar_label', unit(target_col))
    cbar_min = cbar_kws.pop('cbar_min', df[target_col].min() if df[target_col].min() > 0.0 else 1.)
    cbar_max = cbar_kws.pop('cbar_max', df[target_col].max())
    cmap = cbar_kws.pop('cmap', 'jet')

    # Set the plot_kws
    if c is not None:
        plot_kws = dict(marker='o',
                        s=10,
                        edgecolor=None,
                        linewidths=0.3,
                        alpha=0.9,
                        cmap='jet',
                        vmin=df[c].min(),
                        vmax=df[c].max(),
                        c=df[c]
                        **plot_kws)
    else:
        plot_kws = dict(**plot_kws)

    # Set the figure keywords
    fig_kws = dict(**fig_kws)

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # main plot
    if c is not None:
        sc = ax.scatter(time, df[target_col], **plot_kws)

    else:
        line, = ax.plot(time, df[target_col], **plot_kws)

    # set title
    st_tm, fn_tm = time[0], time[-1]
    ax.set_title(kwargs.get('title', ''))
    # ax.set_title(kwargs.get('title', f'{st_tm.strftime("%Y/%m/%d")} - {fn_tm.strftime("%Y/%m/%d")}'))

    # set ticks
    freq = kwargs.get('freq', '10d')
    tick_time = date_range(st_tm, fn_tm, freq=freq)

    ax.set_xticks(tick_time)
    ax.set_xticklabels(tick_time, size=12)

    if ~set_visible:
        ax.axes.xaxis.set_visible(False)

    # Set the figure keywords
    if cbar:
        cbar_kws = dict(label=r'$dN/dlogD_p\ (\# / cm^{-3})$', **cbar_kws)
        clb = plt.colorbar(ax, pad=0.01, **cbar_kws)

    return ax


@set_figure(fs=12)
def time_series(df):
    time = df.index.copy()
    st_tm, fn_tm = df.index[0], df.index[-1]
    tick_time = date_range(st_tm, fn_tm, freq='10d')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(len(df.index) * 0.02, 6))

    ax1 = sub_timeseries(df,
                         target_col='Extinction',
                         ax=ax1,
                         set_visible=False,
                         plot_kws=dict(color="b", label='Extinction'),
                         )
    ax1 = sub_timeseries(df,
                         target_col='Scattering',
                         ax=ax1,
                         set_visible=False,
                         plot_kws=dict(color="g", label='Scattering'),
                         )
    ax1 = sub_timeseries(df,
                         target_col='Absorption',
                         ax=ax1,
                         set_visible=False,
                         plot_kws=dict(color="r", label='Absorption'),
                         )

    ax1.set_ylabel(r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$')
    ax1.set(ylim=(0., df.Extinction.max() * 1.1), xlim=(st_tm, fn_tm))
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, labelspacing=0.5, handlelength=1)

    # Temp, RH
    ax2 = sub_timeseries(df, 'AT', ax=ax2, set_visible=False, plot_kws=dict(color='r', label=unit('AT')))
    ax2.set(ylim=(df.AT.min() - 2, df.AT.max() + 2), xlim=(st_tm, fn_tm))

    # ax2.tick_params(axis='y', colors=ax2.get_facecolor())
    # ax2.yaxis.label.set_color(ax2.get_facecolor())
    # ax2.spines['left'].set_color(ax2.get_facecolor())

    ax2_2 = ax2.twinx()
    ax2_2 = sub_timeseries(df, 'RH', ax=ax2_2, set_visible=False, plot_kws=dict(color='b', label=unit('RH')))
    ax2_2.set(ylim=(20, 100), xlim=(st_tm, fn_tm))

    ax2_2.tick_params(axis='y', colors=ax2_2.get_facecolor())
    # ax2_2.yaxis.label.set_color(ax2_2.get_facecolor())
    # ax2_2.spines['right'].set_color(ax2_2.get_facecolor())
    # ax2_2.spines['left'].set_color(ax2.get_facecolor())

    scalar_map, colors = color_maker(df.PBLH.values)
    ax3.bar(time, df.VC, color=scalar_map.to_rgba(colors), width=0.0417, edgecolor='None', linewidth=0)
    # ax3.bar(time_, np.where(colors == 0, df_.VC, 0), width=0, color='None', alpha=1, edgecolor='None', linewidth=0, zorder=2)
    ax3.set_ylabel(r'$\bf VC\ (m^2/s)$')
    ax3.set(ylim=(0, df.VC.max() * 1.1), xlim=(st_tm, fn_tm))
    ax3.axes.xaxis.set_visible(False)

    # ax3_1 = inset_axes(ax3,
    #                    width="1%",
    #                    height="100%",
    #                    loc='lower left',
    #                    bbox_to_anchor=(1.1, 0., 1, 1),
    #                    bbox_transform=ax3.transAxes,
    #                    borderpad=0,
    #                    )
    #
    # cbar = plt.colorbar(scalar_map, cax=ax3_1, orientation='vertical',)
    ax3_1 = inset_colorbar(ax3, scalar_map, orientation='vertical', inset_kws=dict(),
                           clb_kws=dict(ticks=[0, 200, 400, 600, 800], label=r'$\bf PBLH (m)$'))

    ax3_2 = ax3.twinx()
    sc_6 = ax3_2.scatter(time, df.WS, c=df.WD, cmap='hsv', marker='o', s=5, alpha=1.0)
    ax3_2.set_ylabel(r'$\bf WS\ (m/s)$')
    ax3_2.set_ylim((-2, df.WS.max() * 1.1))
    ax3_2.set_yticks([0, 2, 4])

    # add colorbar
    ax3_3 = inset_colorbar(ax3, sc_6, orientation='horizontal', inset_kws=dict(),
                           clb_kws=dict(label=r'$\bf WD $'))

    # sc_6 = sub_timeseries(df,
    #                       target_col='PM25',
    #                       c='PM1/PM25',
    #                       ax=ax4,
    #                       plot_kws=dict(color="g", label=r'$\bf PM_{2.5}\ (\mu g/m^3)$'),
    #                       )

    sc_6 = ax4.scatter(time, df.PM25, c=df.PM1 / df.PM25, vmin=0.2, vmax=1, cmap='jet', marker='o', s=5, alpha=1.0)

    ax4.set(ylim=(0, df.PM25.max() * 1.2), xlim=(st_tm, fn_tm))

    tick_time = date_range(st_tm, fn_tm, freq='10d')

    ax4.set_xticks(tick_time)
    ax4.set_xticklabels(tick_time, size=12)
    # add colorbar
    ax4_2 = inset_colorbar(ax4, sc_6, orientation='horizontal', inset_kws=dict(),
                           clb_kws=dict(label=r'$\bf PM_{1}/PM_{2.5} $'))

    # fig.savefig(f'time2_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
    plt.show()
