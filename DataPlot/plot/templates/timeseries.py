import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import Axes
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import date_range
from typing import Literal
from DataPlot.plot.core import *


__all__ = ['timeseries']


def _inset_colorbar(axes_image: ScalarMappable,
                    ax: Axes,
                    orientation: Literal['vertical', 'horizontal'] = 'vertical',
                    **kwargs):
    """

    Parameters
    ----------
    axes_image
    ax
    orientation
    inset_kws
    cbar_kws

    Returns
    -------

    """
    if orientation == 'vertical':
        default_inset_kws = dict(
            width="1%",
            height="100%",
            loc='lower left',
            bbox_to_anchor=(1.02, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

    else:
        default_inset_kws = dict(
            width="35%",
            height="7%",
            loc='lower left',
            bbox_to_anchor=(0.015, 0.85, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

    default_inset_kws.update(kwargs.get('inset_kws', {}))

    # create cax for colorbar
    cax = inset_axes(ax, **default_inset_kws)

    default_cbar_kws = dict(
        label='cbar_label',
        cmap='jet',
    )

    default_cbar_kws.update(kwargs)

    # Set clb_kws
    plt.colorbar(mappable=axes_image, cax=cax, **default_cbar_kws)


def _timeseries(df: pd.DataFrame,
                y: str,
                c: str = None,
                style: Literal['scatter', 'bar'] = 'scatter',
                ax: Axes | None = None,
                set_visible=True,
                **kwargs):

    time = df.index.copy()

    if ax is None:
        fig, ax = plt.subplots(**kwargs.get('fig_kws', {}))

    # Set the plot_kws
    if c is not None and style == 'scatter':  # scatterplot
        default_plot_kws = dict(
            marker='o',
            s=5,
            edgecolor=None,
            linewidths=0.3,
            alpha=0.9,
            cmap='jet',
            vmin=df[c].min(),
            vmax=df[c].max(),
            c=df[c],
        )

        default_plot_kws.update(kwargs.get('plot_kws', {}))

        ax.scatter(time, df[y], **default_plot_kws)

        _inset_colorbar(ax.get_children()[0], ax, **kwargs.get('cbar_kws', {}))

    elif c is not None and style == 'bar':  # bar
        default_plot_kws = dict(
            width=0.0417,
            edgecolor=None,
            linewidth=0,
            cmap='jet',
        )

        default_plot_kws.update(kwargs.get('plot_kws', {}))

        scalar_map, colors = Color.color_maker(df[c].values, cmap=default_plot_kws.pop('cmap'))

        ax.bar(time, df[y], color=scalar_map.to_rgba(colors), **default_plot_kws)

        _inset_colorbar(scalar_map, ax, **kwargs.get('cbar_kws', {}))

    else:  # line plot
        ax.plot(time, df[y], **kwargs.get('plot_kws', {}))

    if not set_visible:
        ax.axes.xaxis.set_visible(False)

    if kwargs is not None:
        st_tm, fn_tm = time[0], time[-1]
        freq = kwargs.get('freq', '10d')
        tick_time = date_range(st_tm, fn_tm, freq=freq)

        ax.set(
            xlabel=kwargs.get('xlabel', ''),
            ylabel=kwargs.get('ylabel', Unit(y)),
            xticks=kwargs.get('xticks', tick_time),
            xticklabels=kwargs.get('xticklabels', [_tm.strftime("%F") for _tm in tick_time]),
            xlim=kwargs.get('xlim', (st_tm, fn_tm)),
            ylim=kwargs.get('ylim', (None, None)),
        )

    return ax


@set_figure(fs=10)
def timeseries(df: pd.DataFrame, rolling: int | None = None) -> Axes:

    if rolling is not None:
        df = df.rolling(rolling).mean(numeric_only=True)

    fig, ax = plt.subplots(5, 1, figsize=(len(df.index) * 0.02, 6))
    (ax1, ax2, ax3, ax4, ax5) = ax
    _timeseries(df,
                y='Extinction',
                ax=ax1,
                set_visible=False,
                plot_kws=dict(color="b", label='Extinction'),
                ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                ylim=[0., df.Extinction.max() * 1.1]
                )

    _timeseries(df,
                y='Scattering',
                ax=ax1,
                set_visible=False,
                plot_kws=dict(color="g", label='Scattering'),
                ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                ylim=[0., df.Extinction.max() * 1.1]
                )

    _timeseries(df,
                y='Absorption',
                ax=ax1,
                set_visible=False,
                plot_kws=dict(color="r", label='Absorption'),
                ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                ylim=[0., df.Extinction.max() * 1.1]
                )

    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3, labelspacing=0.5, handlelength=1)

    # Temp, RH
    _timeseries(df,
                y='AT',
                ax=ax2,
                set_visible=False,
                plot_kws=dict(color='r', label=Unit('AT')),
                ylabel=Unit('AT'),
                ylim=[df.AT.min() - 2, df.AT.max() + 2])

    _timeseries(df,
                y='RH',
                ax=ax2.twinx(),
                set_visible=False,
                plot_kws=dict(color='b', label=Unit('RH')),
                ylabel=Unit('RH'),
                ylim=[20, 100])

    _timeseries(df,
                y='VC',
                c='PBLH',
                style='bar',
                ax=ax3,
                set_visible=False,
                plot_kws=dict(label=Unit('VC')),
                cbar_kws=dict(label=Unit('PBLH'))
                )

    _timeseries(df,
                y='WS',
                c='WD',
                ax=ax4,
                set_visible=False,
                plot_kws=dict(cmap='hsv', label=Unit('WS')),
                cbar_kws=dict(label=Unit('WD')),
                ylim=[0, df.WS.max() * 1.1]
                )

    _timeseries(df,
                y='PM25',
                c='PM1/PM25',
                ax=ax5,
                plot_kws=dict(label=Unit('PM1/PM25')),
                cbar_kws=dict(label=Unit('PM1/PM25')),
                ylim=[0, df.PM25.max() * 1.1]
                )

    # fig.savefig(f'tms_{st_tm.strftime("%F")}_{fn_tm.strftime("%F")}.png')

    return ax