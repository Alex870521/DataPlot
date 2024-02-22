import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import date_range
from typing import Literal
from DataPlot.plot.core import unit, set_figure, color_maker


__all__ = ['timeseries',
           'tms_scatter',
           'tms_plot',
           'tms_bar',
           'inset_colorbar']


def _inset_axes(ax: plt.Axes,
                orientation: Literal['vertical', 'horizontal'] = 'vertical',
                inset_kws={}):

    default_inset_kws = {}
    if orientation == 'vertical':
        default_inset_kws = dict(
            width="1%",
            height="100%",
            loc='lower left',
            bbox_to_anchor=(1.02, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

    if orientation == 'horizontal':
        default_inset_kws = dict(
            width="35%",
            height="7%",
            loc='lower left',
            bbox_to_anchor=(0.015, 0.85, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

    default_inset_kws.update(inset_kws)

    return inset_axes(ax, **default_inset_kws)


def inset_colorbar(axes_image: ScalarMappable,
                   ax: plt.Axes,
                   orientation: Literal['vertical', 'horizontal'] = 'vertical',
                   inset_kws={},
                   cbar_kws={}):

    # create cax for colorbar
    cax = _inset_axes(ax, orientation=orientation, inset_kws=inset_kws)

    default_cbar_kws = dict(
        label='cbar_label',
        cmap='jet',
    )

    default_cbar_kws.update(cbar_kws)

    # Set clb_kws
    plt.colorbar(mappable=axes_image, cax=cax, **default_cbar_kws)


def timeseries(df: pd.DataFrame,
               y: str,
               c=None,
               ax=None,
               set_visible=False,
               fig_kws={},
               plot_kws={},
               cbar=False,
               cbar_kws={},
               **kwargs):

    time = df.index.copy()

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # Set the plot_kws
    default_plot_kws = {}
    if c is not None:  # scatterplot
        default_plot_kws = dict(
            marker='o',
            s=10,
            edgecolor=None,
            linewidths=0.3,
            alpha=0.9,
            cmap='jet',
            vmin=df[c].min(),
            vmax=df[c].max(),
            c=df[c],
        )

        default_plot_kws.update(plot_kws)

        ax.scatter(time, df[y], **default_plot_kws)

    else:
        default_plot_kws.update(plot_kws)
        ax.plot(time, df[y], **default_plot_kws)

    if not set_visible:
        ax.axes.xaxis.set_visible(False)

    if cbar:
        inset_colorbar(ax.get_children()[0], ax, cbar_kws=cbar_kws)

    if kwargs is not None:
        st_tm, fn_tm = time[0], time[-1]
        freq = kwargs.get('freq', '10d')
        tick_time = date_range(st_tm, fn_tm, freq=freq)

        ax.set(
            xlabel=kwargs.get('xlabel', ''),
            ylabel=kwargs.get('ylabel', unit(f'{y}')),
            xticks=kwargs.get('xticks', tick_time),
            xticklabels=kwargs.get('xticklabels', [_tm.strftime("%F") for _tm in tick_time]),
            # yticks=kwargs.get('yticks', ''),
            # yticklabels=kwargs.get('yticklabels', ''),
            xlim=kwargs.get('xlim', [st_tm, fn_tm]),
            ylim=kwargs.get('ylim', [None, None]),
        )

    return ax


def tms_scatter(): pass


def tms_plot(): pass


def tms_bar(df,
            y,
            c=None,
            ax=None,
            set_visible=False,
            fig_kws={},
            plot_kws={},
            cbar=False,
            cbar_kws={},
            **kwargs):

    time = df.index.copy()

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # Set the plot_kws
    default_plot_kws = dict(
        width=0.0417,
        edgecolor=None,
        linewidth=0,
        cmap='jet',
    )

    default_plot_kws.update(plot_kws)

    scalar_map, colors = color_maker(df[f'{c}'].values, cmap=default_plot_kws.pop('cmap'))
    ax.bar(time, df[f'{y}'], color=scalar_map.to_rgba(colors), width=0.0417, edgecolor='None', linewidth=0)

    if not set_visible:
        ax.axes.xaxis.set_visible(False)

    if cbar:
        inset_colorbar(scalar_map, ax, cbar_kws=cbar_kws)

    if kwargs is not None:
        st_tm, fn_tm = time[0], time[-1]
        freq = kwargs.get('freq', '10d')
        tick_time = date_range(st_tm, fn_tm, freq=freq)

        ax.set(
            xlabel=kwargs.get('xlabel', ''),
            ylabel=kwargs.get('ylabel', unit(f'{y}')),
            xticks=kwargs.get('xticks', tick_time),
            xticklabels=kwargs.get('xticklabels', [_tm.strftime('%F') for _tm in tick_time]),
            # yticks=kwargs.get('yticks', ''),
            # yticklabels=kwargs.get('yticklabels', ''),
            xlim=kwargs.get('xlim', [st_tm, fn_tm]),
            ylim=kwargs.get('ylim', [None, None]),
        )
