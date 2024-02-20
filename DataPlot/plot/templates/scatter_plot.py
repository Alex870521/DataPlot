import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
from DataPlot.plot import set_figure, unit, getColor, linecolor
from sklearn.linear_model import LinearRegression

# ref https://seaborn.pydata.org/generated/seaborn.scatterplot.html

__all__ = ['scatter',
           'scatter_multiple_reg',
           ]


def _range(series, **kwargs):
    _data = np.array(series)
    max_value, min_value = _data.max() * 1.1, _data.min() * 0.9
    data_range = (min_value, max_value)
    range_from_kwargs = kwargs.get('range')
    return _data, max_value, min_value, range_from_kwargs or data_range


@set_figure(figsize=(6, 5))
def scatter(_df, x, y, c=None, s=None, cmap='jet', regression=None, diagonal=False, box=False, **kwargs):

    df = _df.dropna(subset=[x, y])
    x_data, x_max, x_min, x_range = _range(df[x], range=kwargs.get('x_range'))
    y_data, y_max, y_min, y_range = _range(df[y], range=kwargs.get('y_range'))

    fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize'))  # None = reParams

    if c is not None and s is not None:
        c_data, c_max, c_min, c_range = _range(df[c], range=kwargs.get('c_range'))
        s_data, s_max, s_min, s_range = _range(df[s], range=kwargs.get('s_range'))

        scatter = ax.scatter(x_data, y_data, c=c_data, vmin=c_range[0], vmax=c_range[1], cmap=cmap, s=s_data, alpha=0.7, edgecolors=None)
        colorbar = True

        dot = np.linspace(s_range[0], s_range[1], 6).round(-1)

        for dott in dot[1:-1]:
            plt.scatter([], [], c='k', alpha=0.8, s=300 * (dott / s_data.max()) ** 1.5, label='{:.0f}'.format(dott))

        plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5, title=unit(s))

    elif c is not None:
        c_data, c_max, c_min, c_range = _range(df[c], range=kwargs.get('c_range'))

        scatter = ax.scatter(x_data, y_data, c=c_data, vmin=c_range[0], vmax=c_range[1], cmap=cmap, alpha=0.7, edgecolors=None)
        colorbar = True

    elif s is not None:
        s_data, s_max, s_min, s_range = _range(df[s], range=kwargs.get('s_range'))

        scatter = ax.scatter(x_data, y_data, s=s_data, color='#7a97c9', alpha=0.7, edgecolors='white')
        colorbar = False

        # dealing
        dot = np.linspace(s_range[0], s_range[1], 6).round(-1)

        for dott in dot[1:-1]:
            plt.scatter([], [], c='k', alpha=0.8, s=300 * (dott / s_data.max()) ** 1.5, label='{:.0f}'.format(dott))

        plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5, title=unit(s))

    else:
        scatter = ax.scatter(x_data, y_data, s=30, color='#7a97c9', alpha=0.7, edgecolors='white')
        colorbar = False

    xlim = kwargs.get('xlim') or x_range
    ylim = kwargs.get('ylim') or y_range
    xlabel = kwargs.get('xlabel') or unit(x) or 'xlabel'
    ylabel = kwargs.get('ylabel') or unit(y) or 'ylabel'
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    title = kwargs.get('title') or 'title'
    ax.set_title(title, fontdict={'fontweight': 'bold', 'fontsize': 20})

    # color_bar
    if colorbar:
        color_bar = plt.colorbar(scatter, extend='both')
        color_bar.set_label(label=unit(c) or 'clabel', size=14)

    if regression:
        text = _regression(x_data, y_data, color=sns.xkcd_rgb["denim blue"])

        plt.text(0.05, 0.95, f'{text}', fontdict={'weight': 'bold'}, color=sns.xkcd_rgb["denim blue"],
                 ha='left', va='top', transform=ax.transAxes)

    if diagonal:
        ax.axline((0, 0), slope=1., color='k', lw=2, ls='--', alpha=0.5, label='1:1')
        plt.text(0.91, 0.97, r'$\bf 1:1\ Line$', color='k', ha='right', va='top', transform=ax.transAxes)

    if box:
        bins = np.linspace(x_range[0], x_range[1], 11, endpoint=True)
        wid = (bins + (bins[1] - bins[0]) / 2)[0:-1]

        df[f'{x}' + '_bin'] = pd.cut(x=x_data, bins=bins, labels=wid)

        group = f'{x}' + '_bin'
        column = f'{y}'
        grouped = df.groupby(group)

        names, vals = [], []

        for i, (name, subdf) in enumerate(grouped):
            names.append('{:.0f}'.format(name))
            vals.append(subdf[column].dropna().values)

        plt.boxplot(vals, labels=names, positions=wid, widths=(bins[1] - bins[0]) / 3,
                    showfliers=False, showmeans=True, meanline=True, patch_artist=True,
                    boxprops=dict(facecolor='#f2c872', alpha=.7),
                    meanprops=dict(color='#000000', ls='none'),
                    medianprops=dict(ls='-', color='#000000'))

        plt.xlim(x_range[0], x_range[1])
        ax.set_xticks(bins, labels=bins.astype(int))

    ax.ticklabel_format(axis='both', style='sci', scilimits=(-1, 3), useMathText=True)
    # savefig

    return fig, ax


def _regression(x_data, y_data, color=sns.xkcd_rgb["denim blue"]):
    x_fit = x_data.reshape(-1, 1)
    y_fit = y_data.reshape(-1, 1)

    model = LinearRegression().fit(x_fit, y_fit)

    slope = round(model.coef_[0][0], 3)
    intercept = round(model.intercept_[0], 3)
    r_square = round(model.score(x_fit, y_fit), 3)

    plt.plot(x_fit, model.predict(x_fit), linewidth=3, color=color, alpha=1, zorder=3)

    text = np.poly1d([slope, intercept])
    text = 'y = ' + str(text).replace('\n', "") + '\n' + r'$\bf R^2 = $' + str(r_square)

    return text


@set_figure(figsize=(6, 5))
def scatter_multiple_reg(df, x: str, y: Union[str, List[str]] = None, labels: Union[str, List[str]] = None, ax=None, regression=None, diagonal=False, **kwargs):
    """
    Create a scatter plot with multiple regression lines for the given data.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the data.
    x : str
        Column name for the x-axis variable.
    y : list of str or str
        Column name(s) for the y-axis variable(s). Can be a single string or a list of strings.
    labels : list of str or str, optional
        Labels for the y-axis variable(s). If None, column names are used as labels. Default is None.
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot to use for the plot. If None, a new subplot is created. Default is None.
    regression : bool, optional
        If True, regression lines are plotted for each y variable. Default is None.
    diagonal : bool, optional
        If True, a diagonal line (1:1 line) is added to the plot. Default is False.
    **kwargs
        Additional keyword arguments to customize the plot.

    Returns
    -------
    AxesSubplot
        Matplotlib AxesSubplot containing the scatter plot.

    Notes
    -----
    - The function creates a scatter plot with the option to include multiple regression lines.
    - If regression is True, regression lines are fitted for each y variable.
    - Additional customization can be done using the **kwargs.

    Example
    -------
    >>> scatter_multiple_reg(df, x='X', y=['Y1', 'Y2'], labels=['Label1', 'Label2'],
    ...                      regression=True, diagonal=True, xlim=(0, 10), ylim=(0, 20),
    ...                      xlabel="X-axis", ylabel="Y-axis", title="Scatter Plot with Regressions")
    """

    print('Plot: scatter_multiple_reg')
    if ax is None:
        fig, ax = plt.subplots()

    if not isinstance(y, list):
        y = [y]

    if labels is None:
        labels = y

    df = df.dropna(subset=[x, *y])
    x_data = np.array(df[x])

    color_cycle = linecolor()

    handles = []
    text_list = []

    for i, y_var in enumerate(y):
        y_data = np.array(df[y_var])
        color = color_cycle[i % len(color_cycle)]

        scatter = ax.scatter(x_data, y_data, s=25, color=color['face'], edgecolors=color['edge'], alpha=0.8, label=f'{labels[i]}')
        handles.append(scatter)

        if regression:
            text = _regression(x_data, y_data, color=color['line'])
            text_list.append(f'{labels[i]}: {text}')

    xlim = kwargs.get('xlim')
    ylim = kwargs.get('ylim')
    xlabel = kwargs.get('xlabel') or unit(x) or ''
    ylabel = kwargs.get('ylabel') or unit(y[0]) or ''  # Assuming all y variables have the same unit

    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    title = kwargs.get('title') or ''
    ax.set_title(title, fontdict={'fontweight': 'bold', 'fontsize': 20})

    if regression:
        # Add regression info to the legend
        leg = plt.legend(handles=handles, labels=text_list, loc='upper left', prop={'weight': 'bold', 'size': 10})
    else:
        # Use default legend without regression info
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', prop={'weight': 'bold', 'size': 10})

    for text, color in zip(leg.get_texts(), [color['line'] for color in color_cycle]):
        text.set_color(color)

    if diagonal:
        ax.axline((0, 0), slope=1., color='k', lw=2, ls='--', alpha=0.5, label='1:1')
        plt.text(0.97, 0.97, r'$\bf 1:1\ Line$', color='k', ha='right', va='top', transform=ax.transAxes)

    return ax
