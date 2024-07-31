from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import DataFrame, date_range, Timestamp

from DataPlot.plot.core import *

default_bar_kws = dict(
    width=0.0417,
    edgecolor=None,
    linewidth=0,
    cmap='jet',
)

default_scatter_kws = dict(
    marker='o',
    s=5,
    edgecolor=None,
    linewidths=0.3,
    alpha=0.9,
    cmap='jet',
)

default_insert_kws = dict(
    width="1.5%",
    height="100%",
    loc='lower left',
    bbox_to_anchor=(1.01, 0, 1.2, 1),
    borderpad=0
)


def combine_legends(axes_list: list[Axes]) -> tuple[list, list]:
    return (
        [legend for axes in axes_list for legend in axes.get_legend_handles_labels()[0]],
        [label for axes in axes_list for label in axes.get_legend_handles_labels()[1]]
    )


@set_figure(fs=8, autolayout=False)
def timeseries(df: DataFrame,
               y: list[str] | str,
               y2: list[str] | str = None,
               c: str = None,
               color: list[str] | str = None,
               rolling: str | int | None = None,
               times: tuple[datetime, datetime] | tuple[Timestamp, Timestamp] = None,
               freq: str = '2MS',
               style: Literal['scatter', 'bar', 'line'] | None = 'scatter',
               fill_between: bool = False,
               ax: Axes | None = None,
               set_xaxis_visible: bool | None = None,
               legend_loc: Literal['best', 'upper right', 'upper left', 'lower left', 'lower right'] = 'best',
               legend_ncol: int = 1,
               **kwargs):
    """
    Plot the timeseries data with the option of scatterplot, barplot, and lineplot.

    Parameters
    -----------
    df : DataFrame
    The data to plot.
    y : list[str] | str
        The primary y-axis data columns.
    y2 : list[str] | str, optional
        The secondary y-axis data columns. Defaults to None.
    c : str, optional
        The column for color mapping. Defaults to None.
    color : list[str] | str, optional
        Colors for the plots. Defaults to None.
    rolling : str | int | None, optional
        Rolling window size for smoothing. Defaults to None.
    times : tuple[datetime, datetime] | tuple[Timestamp, Timestamp], optional
        Time range for the data. Defaults to None.
    freq : str, optional
        Frequency for x-axis ticks. Defaults to '2MS'.
    orientation : Literal['vertical', 'horizontal'], optional
        Orientation of the plot. Defaults to 'vertical'.
    style : Literal['scatter', 'bar', 'line'] | None, optional
        Style of the plot. Defaults to 'scatter'.
    fill_between : bool, optional
        Whether to fill between lines for line plots. Defaults to False.
    ax : Axes | None, optional
        Matplotlib Axes object to plot on. Defaults to None.
    set_xaxis_visible : bool | None, optional
        Whether to set x-axis visibility. Defaults to None.
    legend_loc : Literal['best', 'upper right', 'upper left', 'lower left', 'lower right'], optional
        Location of the legend. Defaults to 'best'.
    legend_ncol : int, optional
        Number of columns in the legend. Defaults to 1.
    **kwargs : Additional keyword arguments for customization.
        fig_kws : dict, optional
            Additional keyword arguments for the figure. Defaults to {}.
        scatter_kws : dict, optional
            Additional keyword arguments for the scatter plot. Defaults to {}.
        bar_kws : dict, optional
            Additional keyword arguments for the bar plot. Defaults to {}.
        ax_plot_kws : dict, optional
            Additional keyword arguments for the primary y-axis plot. Defaults to {}.
        ax2_plot_kws : dict, optional
            Additional keyword arguments for the secondary y-axis plot. Defaults to {}.
        cbar_kws : dict, optional
            Additional keyword arguments for the colorbar. Defaults to {}.
        inset_kws : dict, optional
            Additional keyword arguments for the inset axes. Defaults to {}.

    Returns
    -------
    ax : AxesSubplot
        Matplotlib AxesSubplot.

    Example
    -------
    >>> timeseries(df, y='WS', c='WD', scatter_kws=dict(cmap='hsv'), cbar_kws=dict(ticks=[0, 90, 180, 270, 360]), ylim=[0, None])
    """

    # Rolling data
    df = df.rolling(window=rolling, min_periods=1).mean(numeric_only=True) if rolling is not None else df
    df_std = df.rolling(window=rolling, min_periods=1).std(numeric_only=True) if rolling is not None else None

    # Set the time
    if times is None:
        st_tm, fn_tm = df.index[0], df.index[-1]

    else:
        if all(isinstance(t, datetime) for t in times):
            st_tm, fn_tm = Timestamp(times[0]), Timestamp(times[1])
        elif all(isinstance(t, Timestamp) for t in times):
            st_tm, fn_tm = times[0], times[1]
        else:
            raise ValueError('The time should be datetime or Timestamp')

        df = df.loc[st_tm:fn_tm]
        df_std = df_std.loc[st_tm:fn_tm] if df_std is not None else None

    if ax is None:
        default_fig_kws = {**{'figsize': (6, 2)}, **kwargs.get('fig_kws', {})}
        fig, ax = plt.subplots(**default_fig_kws)
    else:
        fig, ax = ax.get_figure(), ax

    # config the y and y2
    y = [y] if isinstance(y, str) else y
    y2 = [y2] if isinstance(y2, str) else y2 if y2 is not None else []

    # Ensure color is a list and check the length
    if color is not None:
        color = [color] if isinstance(color, str) else color
        if len(color) != len(y) + len(y2):
            raise ValueError("The length of color must match the combined length of y and y2")

    # Set color cycle
    ax.set_prop_cycle(color if color is not None else Color.color_cycle)

    if y2:
        ax2 = ax.twinx()
        ax2.set_prop_cycle(color[len(y):] if color is not None else Color.color_cycle[len(y):])

    # Set the plot_kws
    if c is not None and style == 'scatter':  # scatterplot
        default_scatter_kws.update(kwargs.get('scatter_kws', {}))
        default_cbar_kws = {**{'label': Unit(c), 'ticks': None}, **kwargs.get('cbar_kws', {})}
        default_inset_kws = {**default_insert_kws, **{'bbox_transform': ax.transAxes}, **kwargs.get('inset_kws', {})}

        ax.scatter(df.index, df[y], c=df[c], **default_scatter_kws)
        cax = inset_axes(ax, **default_inset_kws)
        plt.colorbar(mappable=ax.get_children()[0], cax=cax, **default_cbar_kws)

    elif c is not None and style == 'bar':  # barplot
        default_bar_kws.update(kwargs.get('bar_kws', {}))
        default_cbar_kws = {**{'label': Unit(c), 'ticks': None}, **kwargs.get('cbar_kws', {})}
        default_inset_kws = {**default_insert_kws, **{'bbox_transform': ax.transAxes}, **kwargs.get('inset_kws', {})}

        scalar_map, colors = Color.color_maker(df[c].values, cmap=default_bar_kws.pop('cmap'))
        ax.bar(df.index, df[y[0]], color=scalar_map.to_rgba(colors), **default_bar_kws)
        cax = inset_axes(ax, **default_inset_kws)
        plt.colorbar(mappable=scalar_map, cax=cax, **default_cbar_kws)

    else:  # line plot
        for i, _y in enumerate(y):
            default_plot_kws = {**{'label': Unit(_y)}, **kwargs.get('ax_plot_kws', {})}

            ax.plot(df.index, df[_y], **default_plot_kws)
            if fill_between:
                ax.fill_between(df.index, df[_y] - df_std[_y], df[_y] + df_std[_y], alpha=0.2, edgecolor=None)

        if y2:
            for i, _y in enumerate(y2):
                default_plot_kws = {**{'label': Unit(_y)}, **kwargs.get('ax2_plot_kws', {})}

                ax2.plot(df.index, df[_y], **default_plot_kws)
                if fill_between:
                    ax2.fill_between(df.index, df[_y] - df_std[_y], df[_y] + df_std[_y], alpha=0.2, edgecolor=None)

                # Combine legends from ax and ax2
                legends_combined, labels_combined = combine_legends([ax, ax2])
                ax.legend(legends_combined, labels_combined, loc=legend_loc, ncol=legend_ncol)

        else:
            ax.legend(loc=legend_loc, ncol=legend_ncol)

    if set_xaxis_visible is not None:
        ax.axes.xaxis.set_visible(set_xaxis_visible)

    ax.set(xlabel=kwargs.get('xlabel', ''),
           ylabel=kwargs.get('ylabel', Unit(y) if isinstance(y, str) else Unit(y[0])),
           xticks=kwargs.get('xticks', date_range(start=st_tm, end=fn_tm, freq=freq).strftime("%F")),
           yticks=kwargs.get('yticks', ax.get_yticks()),
           xticklabels=kwargs.get('xticklabels', date_range(start=st_tm, end=fn_tm, freq=freq).strftime("%F")),
           yticklabels=kwargs.get('yticklabels', ax.get_yticklabels()),
           xlim=kwargs.get('xlim', (st_tm, fn_tm)),
           ylim=kwargs.get('ylim', (None, None)),
           title=kwargs.get('title', '')
           )

    if y2:
        ax2.set(ylabel=kwargs.get('ylabel2', Unit(y2) if isinstance(y2, str) else Unit(y2[0])),
                yticks=kwargs.get('yticks2', ax2.get_yticks()),
                yticklabels=kwargs.get('yticklabels2', ax2.get_yticklabels()),
                ylim=kwargs.get('ylim2', (None, None)))

    return ax


if __name__ == '__main__':
    from DataPlot import *

    df = DataBase('/Users/chanchihyu/NTU/2020能見度計畫/data/All_data.csv')

    # plot.timeseries(df, y=['Extinction'], y2=['PM1'], ylim=[0, None], ylim2=[0, None], rolling=50, legend_loc='upper left', legend_ncol=2, fill_between=True)

    timeseries(df, y='WS', c='WD', scatter_kws=dict(cmap='hsv'), cbar_kws=dict(ticks=[0, 90, 180, 270, 360]),
               ylim=[0, None])
