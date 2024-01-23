import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import date_range
from DataPlot.plot.core import *


def inset_colorbar(axes_image, ax, inset_kws={}, cbar_kws={}):
    # Set inset_axes_kws
    orientation = cbar_kws.pop('orientation', 'vertical')

    default_inset_kws = {}
    default_cbar_kws = {}

    if orientation == 'horizontal':
        default_inset_kws.update({
            'width': "35%",
            'height': "7%",
            'loc': 'lower left',
            'bbox_to_anchor': (0.015, 0.85, 1, 1),
            'bbox_transform': ax.transAxes,
            'borderpad': 0,
        })
    elif orientation == 'vertical':
        default_inset_kws.update({
            'width': "1%",
            'height': "100%",
            'loc': 'lower left',
            'bbox_to_anchor': (1.02, 0., 1, 1),
            'bbox_transform': ax.transAxes,
            'borderpad': 0,
        })

    default_inset_kws.update(inset_kws)
    default_cbar_kws.update(cbar_kws)

    # Set clb_kws
    cax = inset_axes(ax, **default_inset_kws)
    plt.colorbar(mappable=axes_image, cax=cax, orientation=orientation, **default_cbar_kws)


def timeseries(df,
               y,
               c=None,
               ax=None,
               set_visible=False,
               fig_kws=None,
               plot_kws=None,
               cbar=False,
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
        default_plot_kws = dict(**plot_kws)

    # Set the figure keywords
    fig_kws = dict(**fig_kws)

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # main plot
    if c is not None:
        ax.scatter(x, df[y], **default_plot_kws)

    else:
        ax.plot(x, df[y], **default_plot_kws)

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

    if cbar:
        inset_colorbar(ax.get_children()[0], ax, inset_kws=dict(), cbar_kws=cbar_kws)

    # # Set cbar_kws
    # cbar_label = cbar_kws.pop('cbar_label', unit(y))
    # cbar_min = cbar_kws.pop('cbar_min', df[y].min() if df[y].min() > 0.0 else 1.)
    # cbar_max = cbar_kws.pop('cbar_max', df[y].max())
    # cmap = cbar_kws.pop('cmap', 'jet')


def tms_scatter(): pass


def tms_plot(): pass
