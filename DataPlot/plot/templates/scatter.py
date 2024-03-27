import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
from DataPlot.plot.core import set_figure, Color, Unit


def _linear_regression(x_array: np.ndarray,
                       y_array: np.ndarray,
                       columns: str | list[str] | None = None,
                       positive: bool = True):

    if len(x_array.shape) > 1 and x_array.shape[1] >= 2:
        model = LinearRegression(positive=positive).fit(x_array, y_array)

        coefficients = model.coef_[0].round(3)
        intercept = model.intercept_[0].round(3)
        r_square = model.score(x_array, y_array).round(3)
        y_predict = model.predict(x_array)

        equation = ' + '.join([f'{coeff:.3f} * {col}' for coeff, col in zip(coefficients, columns)])
        equation = equation.replace(' + 0.000 * Const', '')  # Remove terms with coefficient 0

        text = 'y = ' + str(equation) + '\n' + r'$\bf R^2 = $' + str(r_square)
        tab = tabulate([[*coefficients, intercept, r_square]], headers=[*columns, 'intercept', 'R^2'], floatfmt=".3f", tablefmt="fancy_grid")
        print('\n' + tab)

        return text, y_predict, coefficients

    else:
        x_array = x_array.reshape(-1, 1)
        y_array = y_array.reshape(-1, 1)

        model = LinearRegression(positive=positive).fit(x_array, y_array)

        slope = model.coef_[0][0].round(3)
        intercept = model.intercept_[0].round(3)
        r_square = model.score(x_array, y_array).round(3)
        y_predict = model.predict(x_array)

        text = np.poly1d([slope, intercept])
        text = 'y = ' + str(text).replace('\n', "") + '\n' + r'$\bf R^2 = $' + str(r_square)

        tab = tabulate([[slope, intercept, r_square]], headers=['slope', 'intercept', 'R^2'], floatfmt=".3f",
                       tablefmt="fancy_grid")
        print('\n' + tab)

        return text, y_predict, slope


@set_figure
def linear_regression(df: pd.DataFrame,
                      x: str | list[str],
                      y: str | list[str],
                      labels: str | list[str] = None,
                      ax: Axes | None = None,
                      diagonal=False,
                      **kwargs) -> Axes:
    """
    Create a scatter plot with multiple regression lines for the given data.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the data.
    x : str or list of str
        Column name(s) for the x-axis variable(s).
    y : str or list of str
        Column name(s) for the y-axis variable(s).
    labels : str or list of str, optional
        Labels for the y-axis variable(s). If None, column names are used as labels. Default is None.
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot to use for the plot. If None, a new subplot is created. Default is None.
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
    >>> linear_regression(df, x='X', y=['Y1', 'Y2'], labels=['Label1', 'Label2'],
    ...                      regression=True, diagonal=True, xlim=(0, 10), ylim=(0, 20),
    ...                      xlabel="X-axis", ylabel="Y-axis", title="Scatter Plot with Regressions")
    """
    if ax is None:
        fig, ax = plt.subplots()

    if not isinstance(x, str):
        x = x[0]

    if not isinstance(y, list):
        y = [y]

    if labels is None:
        labels = y

    df = df.dropna(subset=[x, *y])
    x_array = df[[x]].to_numpy()

    color_cycle = Color.linecolor

    handles = []
    text_list = []

    for i, y_var in enumerate(y):
        y_array = df[[y_var]].to_numpy()
        color = color_cycle[i % len(color_cycle)]

        scatter = ax.scatter(x_array, y_array, s=25, color=color['face'], edgecolors=color['edge'], alpha=0.8,
                             label=labels[i])
        handles.append(scatter)

        text, y_predict, slope = _linear_regression(x_array, y_array)
        text_list.append(f'{labels[i]}: {text}')
        plt.plot(x_array, y_predict, linewidth=3, color=color['line'], alpha=1, zorder=3)

    xlim = kwargs.get('xlim')
    ylim = kwargs.get('ylim')
    xlabel = kwargs.get('xlabel') or Unit(x)
    ylabel = kwargs.get('ylabel') or Unit(y[0])  # Assuming all y variables have the same unit
    title = kwargs.get('title', '')

    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)

    # Add regression info to the legend
    leg = plt.legend(handles=handles, labels=text_list, loc='upper left', prop={'weight': 'bold', 'size': 10})

    for text, color in zip(leg.get_texts(), [color['line'] for color in color_cycle]):
        text.set_color(color)

    if diagonal:
        ax.axline((0, 0), slope=1., color='k', lw=2, ls='--', alpha=0.5, label='1:1')
        plt.text(0.97, 0.97, r'$\bf 1:1\ Line$', color='k', ha='right', va='top', transform=ax.transAxes)

    return ax


@set_figure
def multiple_linear_regression(df: pd.DataFrame,
                               x: str | list[str],
                               y: str | list[str],
                               labels: str | list[str] = None,
                               ax: Axes | None = None,
                               diagonal=False,
                               add_constant=True,
                               **kwargs) -> Axes:
    """
    Perform multiple linear regression analysis and plot the results.

    Parameters
    ----------
    df : pandas.DataFrame
       Input DataFrame containing the data.
    x : str or list of str
       Column name(s) for the independent variable(s). Can be a single string or a list of strings.
    y : str or list of str
       Column name(s) for the dependent variable(s). Can be a single string or a list of strings.
    labels : str or list of str, optional
       Labels for the dependent variable(s). If None, column names are used as labels. Default is None.
    ax : matplotlib.axes.Axes or None, optional
       Matplotlib Axes object to use for the plot. If None, a new subplot is created. Default is None.
    diagonal : bool, optional
       Whether to include a diagonal line (1:1 line) in the plot. Default is False.
    add_constant : bool, optional
       Whether to add a constant term to the independent variables. Default is True.
    **kwargs
       Additional keyword arguments to customize the plot.

    Returns
    -------
    matplotlib.axes.Axes
       Matplotlib Axes object containing the regression plot.

    Notes
    -----
    This function performs multiple linear regression analysis using the input DataFrame.
    It supports multiple independent variables and can plot the regression results.

    Example
    -------
    >>> multiple_linear_regression(df, x=['X1', 'X2'], y='Y', labels=['Y1', 'Y2'],
    ...                             diagonal=True, add_constant=True,
    ...                             xlabel="X-axis", ylabel="Y-axis", title="Multiple Linear Regression Plot")
    """
    if ax is None:
        fig, ax = plt.subplots()

    if not isinstance(x, list):
        x = [x]

    if not isinstance(y, str):
        y = y[0]

    if labels is None:
        labels = x

    if add_constant:
        df = df.assign(Const=1)
        x_array = df[[*x, 'Const']].to_numpy()
        y_array = df[[y]].to_numpy()
        columns = [*x, 'Const']

    else:
        x_array = df[[*x]].to_numpy()
        y_array = df[[y]].to_numpy()
        columns = [*x]

    text, y_predict, coefficients = _linear_regression(x_array, y_array, columns=columns, positive=True)

    df = pd.DataFrame(np.concatenate([y_array, y_predict], axis=1), columns=['y_actual', 'y_predict'])

    linear_regression(df, x='y_actual', y='y_predict', ax=ax, regression=True, diagonal=diagonal)

    return ax


@set_figure
def scatter(df: pd.DataFrame,
            x: str,
            y: str,
            c: str | None = None,
            s: str | None = None,
            cmap='jet',
            regression=None,
            diagonal=False,
            box=False,
            ax: Axes | None = None,
            **kwargs) -> Axes:

    if ax is None:
        fig, ax = plt.subplots(**kwargs.get('fig_kws', {}))

    plt.subplots_adjust(right=0.9, bottom=0.125)

    df = df.dropna(subset=[x, y]).copy()
    x_data, y_data = df[x].to_numpy(), df[y].to_numpy()

    if c is not None and s is not None:
        c_data = df[c].to_numpy()
        s_data = df[s].to_numpy()

        scatter = ax.scatter(x_data, y_data, c=c_data, norm=Normalize(vmin=np.percentile(c_data, 10), vmax=np.percentile(c_data, 90)),
                            cmap=cmap, s=50 * (s_data / s_data.max()) ** 1.5, alpha=0.7, edgecolors=None)
        colorbar = True

        dot = np.linspace(s_data.min(), s_data.max(), 6).round(-1)

        for dott in dot[1:-1]:
            plt.scatter([], [], c='k', alpha=0.8, s=50 * (dott / s_data.max()) ** 1.5, label='{:.0f}'.format(dott))

        plt.legend(title=Unit(s))

    elif c is not None:
        c_data = df[c].to_numpy()

        scatter = ax.scatter(x_data, y_data, c=c_data, vmin=c_data.min(), vmax=np.percentile(c_data, 90), cmap=cmap, alpha=0.7,
                             edgecolors=None)
        colorbar = True

    elif s is not None:
        s_data = df[s].to_numpy()

        scatter = ax.scatter(x_data, y_data, s=50 * (s_data / s_data.max()) ** 1.5, color='#7a97c9', alpha=0.7, edgecolors='white')
        colorbar = False

        # dealing
        dot = np.linspace(s_data.min(), s_data.max(), 6).round(-1)

        for dott in dot[1:-1]:
            plt.scatter([], [], c='k', alpha=0.8, s=50 * (dott / s_data.max()) ** 1.5, label='{:.0f}'.format(dott))

        plt.legend(title=Unit(s))

    else:
        scatter = ax.scatter(x_data, y_data, s=30, color='#7a97c9', alpha=0.7, edgecolors='white')
        colorbar = False

    xlim = kwargs.get('xlim', (x_data.min(), x_data.max()))
    ylim = kwargs.get('ylim', (y_data.min(), y_data.max()))
    xlabel = kwargs.get('xlabel', Unit(x))
    ylabel = kwargs.get('ylabel', Unit(y))
    title = kwargs.get('title', '')
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)

    # color_bar
    if colorbar:
        color_bar = plt.colorbar(scatter, extend='both')
        color_bar.set_label(label=Unit(c), size=14)

    if regression:
        text, y_predict, slope = _linear_regression(x_data, y_data)
        plt.plot(x_data, y_predict, linewidth=3, color=sns.xkcd_rgb["denim blue"], alpha=1, zorder=3)

        plt.text(0.05, 0.95, f'{text}', fontdict={'weight': 'bold'}, color=sns.xkcd_rgb["denim blue"],
                 ha='left', va='top', transform=ax.transAxes)

    if diagonal:
        ax.axline((0, 0), slope=1., color='k', lw=2, ls='--', alpha=0.5, label='1:1')
        plt.text(0.91, 0.97, r'$\bf 1:1\ Line$', color='k', ha='right', va='top', transform=ax.transAxes)

    if box:
        bins = np.linspace(x_data.min(), x_data.max(), 11, endpoint=True)
        wid = (bins + (bins[1] - bins[0]) / 2)[0:-1]

        df[x + '_bin'] = pd.cut(x=x_data, bins=bins, labels=wid)

        group = x + '_bin'
        column = y
        grouped = df.groupby(group, observed=False)

        names, vals = [], []

        for i, (name, subdf) in enumerate(grouped):
            names.append('{:.0f}'.format(name))
            vals.append(subdf[column].dropna().values)

        plt.boxplot(vals, labels=names, positions=wid, widths=(bins[1] - bins[0]) / 3,
                    showfliers=False, showmeans=True, meanline=True, patch_artist=True,
                    boxprops=dict(facecolor='#f2c872', alpha=.7),
                    meanprops=dict(color='#000000', ls='none'),
                    medianprops=dict(ls='-', color='#000000'))

        plt.xlim(x_data.min(), x_data.max())
        ax.set_xticks(bins, labels=bins.astype(int))

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    # savefig

    return ax
