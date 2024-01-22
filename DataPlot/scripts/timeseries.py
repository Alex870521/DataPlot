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
        defult_inset_kws = dict(width="35%",
                                height="7%",
                                loc='lower left',
                                bbox_to_anchor=(0.015, 0.85, 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0, )

        defult_inset_kws.update(inset_kws)

    if orientation == 'vertical':
        defult_inset_kws = dict(width="1%",
                                height="100%",
                                loc='lower left',
                                bbox_to_anchor=(1.02, 0., 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0,
                                )
        defult_inset_kws.update(inset_kws)

    # Set clb_kws
    cax = inset_axes(ax, **defult_inset_kws)
    color_bar = plt.colorbar(axes_image, cax=cax, orientation=orientation, **clb_kws)


def timeseries(df,
               y,
               c=None,
               ax=None,
               cbar=False,
               set_visible=True,
               fig_kws=None,
               plot_kws=None,
               cbar_kws=None,
               **kwargs):

    if cbar_kws is None:
        cbar_kws = {}
    if plot_kws is None:
        plot_kws = {}
    if fig_kws is None:
        fig_kws = {}

    x = df.index.copy()

    # Set the plot_kws

    if c is not None:
        default_plot_kws = dict(marker='o',
                                s=10,
                                edgecolor=None,
                                linewidths=0.3,
                                alpha=0.9,
                                cmap='jet',
                                vmin=df[c].min(),
                                vmax=df[c].max(),
                                c=df[c],
                                )

        default_plot_kws.update(**plot_kws)

    else:
        plot_kws = dict(**plot_kws)

    # Set the figure keywords
    fig_kws = dict(**fig_kws)

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # main plot
    if c is not None:
        sc = ax.scatter(x, df[y], **default_plot_kws)

    else:
        line, = ax.plot(x, df[y], **plot_kws)

    # prepare parameter
    st_tm, fn_tm = x[0], x[-1]
    freq = kwargs.get('freq', '10d')
    tick_time = date_range(st_tm, fn_tm, freq=freq)

    # set
    if kwargs is not None:
        ax.set(
            xlabel=kwargs.get('xlabel', ''),
            ylabel=kwargs.get('ylabel', unit(f'{y}')),
            xticks=kwargs.get('xticks', tick_time),
            xticklabels=kwargs.get('xticklabels', [_tm.strftime('%Y-%m-%d') for _tm in tick_time]),
            # yticks=kwargs.get('yticks', ''),
            # yticklabels=kwargs.get('yticklabels', ''),
            xlim=kwargs.get('xlim', [st_tm, fn_tm]),
            ylim=kwargs.get('ylim', [None, None]),
        )

    if not set_visible:
        ax.axes.xaxis.set_visible(False)

    # # Set cbar_kws
    # cbar_label = cbar_kws.pop('cbar_label', unit(y))
    # cbar_min = cbar_kws.pop('cbar_min', df[y].min() if df[y].min() > 0.0 else 1.)
    # cbar_max = cbar_kws.pop('cbar_max', df[y].max())
    # cmap = cbar_kws.pop('cmap', 'jet')
    #
    # if cbar:
    #     clb = plt.colorbar(ax, pad=0.01, **cbar_kws)

    if c is not None:
        return sc
    else:
        return ax


@set_figure(fs=12)
def time_series(df):
    time = df.index.copy()
    st_tm, fn_tm = time[0], time[-1]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(len(df.index) * 0.02, 6))

    ax1 = timeseries(df,
                     y='Extinction',
                     ax=ax1,
                     set_visible=False,
                     plot_kws=dict(color="b", label='Extinction'),
                     ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                     ylim=[0., df.Extinction.max() * 1.1]
                     )

    ax1 = timeseries(df,
                     y='Scattering',
                     ax=ax1,
                     set_visible=False,
                     plot_kws=dict(color="g", label='Scattering'),
                     ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                     ylim=[0., df.Extinction.max() * 1.1]
                     )

    ax1 = timeseries(df,
                     y='Absorption',
                     ax=ax1,
                     set_visible=False,
                     plot_kws=dict(color="r", label='Absorption'),
                     ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                     ylim=[0., df.Extinction.max() * 1.1]
                     )

    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, labelspacing=0.5, handlelength=1)

    # Temp, RH
    ax2 = timeseries(df,
                     y='AT',
                     ax=ax2,
                     set_visible=False,
                     plot_kws=dict(color='r', label=unit('AT')),
                     ylabel=unit('AT'),
                     ylim=[df.AT.min() - 2, df.AT.max() + 2])

    ax2_2 = ax2.twinx()
    ax2_2 = timeseries(df, 'RH',
                       ax=ax2_2,
                       set_visible=False,
                       plot_kws=dict(color='b', label=unit('RH')),
                       ylabel=unit('RH'),
                       ylim=[20, 100])

    scalar_map, colors = color_maker(df.PBLH.values)
    ax3.bar(time, df.VC, color=scalar_map.to_rgba(colors), width=0.0417, edgecolor='None', linewidth=0)
    ax3.set_ylabel(r'$\bf VC\ (m^2/s)$')
    ax3.set(ylim=(0, df.VC.max() * 1.1), xlim=(st_tm, fn_tm))
    ax3.axes.xaxis.set_visible(False)

    inset_colorbar(ax3, scalar_map, orientation='vertical', inset_kws=dict(),
                   clb_kws=dict(ticks=[0, 200, 400, 600, 800], label=unit('PBLH')))

    sc_1 = timeseries(df,
                      y='WS',
                      c='WD',
                      ax=ax4,
                      set_visible=False,
                      plot_kws=dict(cmap='hsv', label=unit('WS')),
                      ylabel=unit('WS'),
                      ylim=[0, df.WS.max() * 1.1],
                      )

    # add colorbar
    inset_colorbar(ax4, sc_1, orientation='vertical', inset_kws=dict(),
                   clb_kws=dict(ticks=[0, 180, 360], label=r'$\bf WD $'))

    df['PM1/PM25'] = df.PM1 / df.PM25
    sc_2 = timeseries(df,
                      y='PM25',
                      c='PM1/PM25',
                      ax=ax5,
                      set_visible=True,
                      plot_kws=dict(vmin=0.2, vmax=1, cmap='jet', label=unit('PM25')),
                      ylabel=unit('PM25'))

    inset_colorbar(ax5, sc_2, orientation='vertical', inset_kws=dict(),
                   clb_kws=dict(label=unit('PM1/PM25')))

    # fig.savefig(f'time2_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
    plt.show()
