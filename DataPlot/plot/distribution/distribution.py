import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tabulate import tabulate
from numpy import log, exp, sqrt, pi
from pandas import DataFrame, date_range
from typing import Literal
from scipy.stats import norm, lognorm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.pyplot import Figure, Axes
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.collections import PolyCollection
from DataPlot.plot.core import *
from DataPlot.process import *


__all__ = ['heatmap',
           'heatmap_tms',
           'plot_dist',
           'three_dimension',
           'curve_fitting',
           'ls_mode',
           'lognorm_dist',
           ]


@set_figure(fs=12)
def heatmap(data: DataFrame,
            unit: Literal["Number", "Surface", "Volume", "Extinction"] = 'Number',
            cmap: str = 'Blues',
            ax: Axes | None = None,
            **kwargs) -> Axes:
    """
    Plot a heatmap of particle size distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        The data containing particle size distribution values. Each column corresponds to a size bin,
        and each row corresponds to a different distribution.
    unit : {'Number', 'Surface', 'Volume', 'Extinction'}, optional
        The unit of measurement for the data.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the heatmap on. If not provided, a new subplot will be created.
    cmap : str, default='Blues'
        The colormap to use for the heatmap.
    **kwargs
        Additional keyword arguments to pass to matplotlib functions.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the heatmap.

    Examples
    --------
    >>> ax = heatmap(DataFrame(...), unit='Number')

    Notes
    -----
    This function calculates a 2D histogram of the log-transformed particle sizes and the distribution values.
    It then plots the heatmap using a logarithmic color scale.

    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs.get('fig_kws', {}))

    dp = np.array(data.columns, dtype=float)
    x = np.append(np.tile(dp, data.values.shape[0]), 2750)
    y = np.append(data.values.flatten(), 0.000001)

    nan_indices = np.isnan(y)

    limit = np.percentile(y, [90])

    valid_indices = (y < limit) | (~nan_indices)

    x = x[valid_indices]
    y = y[valid_indices]

    # using log(x)
    heatmap, xedges, yedges = np.histogram2d(np.log(x), y, bins=50)
    heatmap[heatmap == 0] = 0.000001  # Avoid log(0)

    plot_kws = dict(norm=colors.LogNorm(vmin=1, vmax=heatmap.max()), cmap=cmap, **kwargs.get('plot_kws', {}))

    surf = ax.pcolormesh(xedges[:-1], yedges[:-1], heatmap.T, shading='gouraud', antialiased=True, **plot_kws)
    ax.plot(np.log(dp), data.mean(), ls='solid', color='k', lw=2, label='mean')
    # ax.plot(np.log(dp), data.mean() + data.std(), ls='dashed', color='k', lw=2, label='mean')
    # ax.plot(np.log(dp), data.mean() - data.std(), ls='dashed', color='k', lw=2, label='mean')

    ax.set_xticks(np.log([10, 100, 1000]))
    ax.set_xticks(np.log([20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000]), minor=True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda tick, pos: "{:.0f}".format(np.exp(tick))))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1, 4), useMathText=True, useLocale=True)

    ax.set(xlim=(np.log(dp).min(), np.log(dp).max()), ylim=(0, None),
           xlabel=r'$D_{p} (nm)$', ylabel=Unit(f'{unit}_dist'), title=kwargs.get('title', unit))

    ax.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)

    # cbar
    cax = inset_axes(ax, width="5%", height="100%", loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)

    plt.colorbar(surf, cax=cax, label='Counts')

    ax.legend(prop=dict(weight='bold'))

    return ax


@set_figure(fs=12)
def heatmap_tms(data: DataFrame,
                logy: bool = True,
                cbar: bool = True,
                unit: Literal["Number", "Surface", "Volume", "Extinction"] = 'Number',
                hide_low: bool = True,
                cmap: str = 'jet',
                ax: Axes | None = None,
                **kwargs) -> Axes:
    """ Plot the size distribution over time.

    Parameters
    ----------
    data : DataFrame
        A DataFrame of particle concentrations to plot the heatmap.
    ax : matplotlib.axis.Axis
        An axis object to plot on. If none is provided, one will be created.
    logy : bool, default=True
        If true, the y-axis will be semilogy.
    cbar : bool, default=True
        If true, a colorbar will be added.
    unit : Literal["Number", "Surface", "Volume", "Extinction"]
        default='Number'
    hide_low : bool, default=True
        If true, low values will be masked.
    cmap : matplotlib.colormap, default='viridis'
        The colormap to use. Can be anything other that 'jet'.

    Returns
    -------
    ax : matplotlib.axis.Axis

    Examples
    --------
    Plot a SPMS + APS data:
    >>> ax = heatmap_tms(DataFrame(...), cmap='jet')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(len(data.index) * 0.02, 3), **kwargs.get('fig_kws', {}))

    fig.subplots_adjust(right=0.99)

    # Copy to avoid modifying original data
    time = data.index
    dp = np.array(data.columns, dtype=float)
    data = data.copy().to_numpy()
    data = np.nan_to_num(data)

    # Increase values below cbar_min to cbar_min
    if hide_low:
        below_min = data == np.NaN
        data[below_min] = np.NaN

    vmin_mapping = {'Number': 1e4, 'Surface': 1e8, 'Volume': 1e9, 'Extinction': 1}

    # Set the colorbar min and max based on the min and max of the values
    cbar_min = kwargs.get('cbar_kws', {}).pop('cbar_min', vmin_mapping[unit])
    cbar_max = kwargs.get('cbar_kws', {}).pop('cbar_max', np.nanmax(data))

    # Set the plot_kws
    plot_kws = dict(norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max), cmap=cmap, **kwargs.get('plot_kws', {}))

    # main plot
    pco1 = ax.pcolormesh(time, dp, data.T, shading='auto', **plot_kws)

    # Set ax
    st_tm, fn_tm = time[0], time[-1]
    freq = kwargs.get('freq', '10d')
    tick_time = date_range(st_tm, fn_tm, freq=freq)

    ax.set(xlabel=kwargs.get('xlabel', ''),
           ylabel=kwargs.get('ylim', r'$\bf D_p\ (nm)$'),
           xticks=kwargs.get('xticks', tick_time),
           xticklabels=kwargs.get('xticklabels', [_tm.strftime("%F") for _tm in tick_time]),
           xlim=kwargs.get('xlim', (st_tm, fn_tm)),
           ylim=kwargs.get('ylim', (dp.min(), dp.max())),
           title=kwargs.get('title', f'{st_tm.strftime("%F")} - {fn_tm.strftime("%F")}')
           )

    # Set the axis to be log in the y-axis
    if logy:
        ax.semilogy()
        ax.yaxis.set_major_formatter(ScalarFormatter())

    # Set the figure keywords
    if cbar:
        cbar_kws = dict(label=Unit(f'{unit}_dist'), **kwargs.get('cbar_kws', {}))
        plt.colorbar(pco1, pad=0.01, **cbar_kws)

    # fig.savefig(f'heatmap_tm_{st_tm.strftime("%F")}_{fn_tm.strftime("%F")}.png')

    return ax


@set_figure
def plot_dist(data: DataFrame | np.ndarray,
              data_std: DataFrame | None = None,
              std_scale: float | None = 1,
              unit: Literal["Number", "Surface", "Volume", "Extinction"] = 'Number',
              additional: Literal["std", "enhancement", "error"] = None,
              fig: Figure | None = None,
              ax: Axes | None = None,
              **kwargs) -> Axes:
    """
    Plot particle size distribution curves and optionally show enhancements.

    Parameters
    ----------
    data : dict or list
        If dict, keys are labels and values are arrays of distribution values.
        If listed, it should contain three arrays for different curves.
    data_std : dict
        Dictionary containing standard deviation data for ambient extinction distribution.
    std_scale : float
        The width of standard deviation.
    unit : {'Number', 'Surface', 'Volume', 'Extinction'}
        Unit of measurement for the data.
    additional : {'std', 'enhancement', 'error'}
        Whether to show enhancement curves.
    fig : Figure, optional
        Matplotlib Figure object to use.
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot object to use. If not provided, a new subplot will be created.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ax : AxesSubplot
        Matplotlib AxesSubplot.

    Examples
    --------
    >>> data={'Clena': np.array([1, 2, 3, 4]), 'Transition': np.array([1, 2, 3, 4]), 'Event': np.array([1, 2, 3, 4])}
    >>> plot_dist(DataFrame.from_dict(data, orient='index', columns=['11.8', '12.18', '12.58', '12.99']), additional="enhancement")
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots(**kwargs.get('fig_kws', {}))

    # plot_kws
    plot_kws = dict(ls='solid', lw=2, alpha=0.8, **kwargs.get('plot_kws', {}))

    # Receive input data
    dp = np.array(data.columns, dtype=float)
    states = np.array(data.index)

    for state in states:
        mean = data.loc[state].values
        ax.plot(dp, mean, label=state, color=Color.color_choose[state][0], **plot_kws)

        if additional == 'std':
            std = data_std.loc[state].values * std_scale
            ax.fill_between(dp, y1=mean - std, y2=mean + std, alpha=0.4, color=Color.color_choose[state][1],
                            edgecolor=None, label='__nolegend__')

    # figure_set
    ax.set(xlim=(dp.min(), dp.max()), ylim=(0, None), xscale='log',
           xlabel=r'$D_{p} (nm)$', ylabel=Unit(f'{unit}_dist'), title=kwargs.get('title', unit))

    ax.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useMathText=True)

    Clean = data.loc['Clean'].values
    Transition = data.loc['Transition'].values
    Event = data.loc['Event'].values

    if additional == "enhancement":
        ax2 = ax.twinx()
        ax2.plot(dp, Transition / Clean, ls='dashed', color='k', label='Enhancement ratio 1')
        ax2.plot(dp, Event / Transition, ls='dashed', color='gray', label='Enhancement ratio 2')
        ax2.set(xlim=ax.get_xlim(), ylim=(0.5, None), xlabel=ax.get_xlabel(), ylabel='Enhancement ratio')

    elif additional == "error":
        ax2 = ax.twinx()
        error1 = np.where(Transition != 0, np.abs(Clean - Transition) / Transition * 100, 0)
        error2 = np.where(Event != 0, np.abs(Transition - Event) / Event * 100, 0)

        ax2.plot(dp, error1, ls='--', color='k', label='Error 1 ')
        ax2.plot(dp, error2, ls='--', color='gray', label='Error 2')
        ax2.set(xlim=ax.get_xlim(), ylim=(None, None), xlabel=ax.get_xlabel(), ylabel='Error (%)')

    # Combine legends from ax and ax2
    axes_list = fig.get_axes()
    legends_combined = [legend for axes in axes_list for legend in axes.get_legend_handles_labels()[0]]
    labels_combined = [label for axes in axes_list for label in axes.get_legend_handles_labels()[1]]

    ax.legend(legends_combined, labels_combined, prop={'weight': 'bold'})

    # fig.savefig(f'dist_{figname}')

    return ax


@set_figure
def three_dimension(data: DataFrame | np.ndarray,
                    unit: Literal["Number", "Surface", "Volume", "Extinction"],
                    ax: Axes | None = None,
                    **kwargs) -> Axes:
    """
    Create a 3D plot with data from a pandas DataFrame or numpy array.

    Parameters
    ----------
    data : DataFrame or ndarray
        Input data containing the values to be plotted.
    unit : {'Number', 'Surface', 'Volume', 'Extinction'}
        Unit of measurement for the data.
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot. If not provided, a new subplot will be created.
    **kwargs
        Additional keyword arguments to customize the plot.

    Returns
    -------
    Axes
        Matplotlib Axes object representing the 3D plot.

    Notes
    -----
    - The function creates a 3D plot with data provided in a pandas DataFrame or numpy array.
    - The x-axis is logarithmically scaled, and ticks and labels are formatted accordingly.
    - Additional customization can be done using the **kwargs.

    Example
    -------
    >>> data = DataFrame(...)
    >>> three_dimension(data, unit='Number', figsize=(6, 6), cmap='Blues')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"}, **kwargs.get('fig_kws', {}))

    lines = data.shape[0]

    dp = np.array(['11.7', *data.columns, '2437.4'], dtype=float)

    _X, _Y = np.meshgrid(np.log(dp), np.arange(lines))
    _Z = np.pad(data, ((0, 0), (1, 1)), 'constant')

    verts = []
    for i in range(_X.shape[0]):
        verts.append(list(zip(_X[i, :], _Z[i, :])))

    facecolors = plt.colormaps['Blues'](np.linspace(0, 1, len(verts)))
    poly = PolyCollection(verts, facecolors=facecolors, edgecolors='k', lw=0.5, alpha=.7)
    ax.add_collection3d(poly, zs=range(1, lines + 1), zdir='y')
    # ax.set_xscale('log') <- dont work
    ax.set(xlim=(np.log(11.7), np.log(2437.4)), ylim=(1, lines), zlim=(0, np.nanmax(_Z)))
    ax.set_xlabel(r'$\bf D_{p}\ (nm)$', labelpad=10)
    ax.set_ylabel(r'$\bf Class$', labelpad=10)
    ax.set_zlabel(Unit(f'{unit}_dist'), labelpad=15)

    major_ticks = np.log([10, 100, 1000])
    minor_ticks = np.log([20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000])
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_major_formatter(FuncFormatter((lambda tick, pos: "{:.0f}".format(np.exp(tick)))))
    ax.zaxis.get_offset_text().set_visible(False)
    exponent = math.floor(math.log10(np.max(data)))
    ax.text(ax.get_xlim()[1] * 1.05, ax.get_ylim()[1], ax.get_zlim()[1] * 1.1, s=fr'${{\times}}\ 10^{exponent}$')
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0), useOffset=False)

    return ax


@set_figure
def curve_fitting(dp: np.ndarray | pd.Series | DataFrame,
                  dist: np.ndarray | pd.Series | DataFrame,
                  mode: int,
                  unit: Literal["Number", "Surface", "Volume", "Extinction"] = 'Number',
                  ax: Axes | None = None,
                  **kwargs) -> Axes:
    """
    Fit a log-normal distribution to the given data and plot the result.

    Parameters
    ----------
    - dp (array): Array of diameter values.
    - dist (array): Array of distribution values corresponding to each diameter.
    - mode (int, optional): Number of log-normal distributions to fit (default is None).
    - **kwargs: Additional keyword arguments to be passed to the plot_function.

    Returns
    -------
    None

    Notes
    -----
    - The function fits a sum of log-normal distributions to the input data.
    - The number of distributions is determined by the 'mode' parameter.
    - Additional plotting customization can be done using the **kwargs.

    Example
    -------
    >>> curve_fitting(dp, dist, mode=2, xlabel="Diameter (nm)", ylabel="Distribution", figname="extinction")
    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs.get('fig_kws', {}))

    # Calculate total number concentration and normalize distribution
    total_num = np.sum(dist * log(dp))
    norm_data = dist / total_num

    def lognorm_func(x, *params):
        num_distributions = len(params) // 3
        result = np.zeros_like(x)

        for i in range(num_distributions):
            offset = i * 3
            _number = params[offset]
            _geomean = params[offset + 1]
            _geostd = params[offset + 2]
            result += (_number / (log(_geostd) * sqrt(2 * pi)) *
                       exp(-(log(x) - log(_geomean)) ** 2 / (2 * log(_geostd) ** 2)))

        return result

    # initial gauss
    min_value = np.array([min(dist)])
    extend_ser = np.concatenate([min_value, dist, min_value])
    _mode, _ = find_peaks(extend_ser, distance=20)
    peak = dp[_mode - 1]
    mode = mode or len(peak)

    # 設定參數範圍
    bounds = ([1e-6, 10, 1] * mode,
              [1, 3000, 8] * mode)

    # 初始參數猜測
    initial_guess = [0.05, 20, 2] * mode

    # 使用 curve_fit 函數進行擬合
    popt, pcov = curve_fit(lognorm_func, dp, norm_data, p0=initial_guess, maxfev=2000000, method='trf', bounds=bounds)

    # 獲取擬合的參數
    params = popt.tolist()

    print('\n' + "Fitting Results:")
    table = []

    for i in range(mode):
        offset = i * 3
        num, mu, sigma = params[offset:offset + 3]
        table.append([f'log-{i + 1}', num * total_num, mu, sigma])

    formatted_data = [[item if not isinstance(item, float) else f"{item:.3f}" for item in row] for row in table]

    # 使用 tabulate 來建立表格並印出
    tab = tabulate(formatted_data, headers=["log-", "number", "mu", "sigma"], floatfmt=".3f", tablefmt="fancy_grid")
    print(tab)

    fit_curve = total_num * lognorm_func(dp, *params)

    plt.plot(dp, fit_curve, color='#c41b1b', label='Fitting curve', lw=2.5)
    plt.plot(dp, dist, color='b', label='Observed curve', lw=2.5)

    ax.set(xlim=(dp.min(), dp.max()), ylim=(0, None), xscale='log',
           xlabel=r'$\bf D_{p}\ (nm)$', ylabel=Unit(f'{unit}_dist'), title=kwargs.get('title', ''))

    plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useMathText=True)
    ax.legend()

    # plt.savefig(f'CurveFit_{figname}.png')

    return ax


@set_figure
def ls_mode(**kwargs) -> Axes:
    """
    Plot log-normal mass size distribution for small mode, large mode, and sea salt particles.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments.

    Examples
    --------
    Example : Plot log-normal mass size distribution with default settings
    >>> ls_mode()
    """

    fig, ax = plt.subplots(**kwargs.get('fig_kws', {}))

    geoMean = [0.2, 0.5, 2.5]
    geoStdv = [2.2, 1.5, 2.0]
    color = ['g', 'r', 'b']
    label = [r'$\bf Small\ mode\ :D_{g}\ =\ 0.2\ \mu m,\ \sigma_{{g}}\ =\ 2.2$',
             r'$\bf Large\ mode\ :D_{g}\ =\ 0.5\ \mu m,\ \sigma_{{g}}\ =\ 1.5$',
             r'$\bf Sea\ salt\ :D_{g}\ =\ 2.5\ \mu m,\ \sigma_{{g}}\ =\ 2.0$']

    x = np.geomspace(0.001, 20, 10000)
    for _gmd, _gsd, _color, _label in zip(geoMean, geoStdv, color, label):
        lognorm = 1 / (log(_gsd) * sqrt(2 * pi)) * (exp(-(log(x) - log(_gmd)) ** 2 / (2 * log(_gsd) ** 2)))

        ax.semilogx(x, lognorm, color=_color, label=_label)
        ax.fill_between(x, lognorm, 0, where=(lognorm > 0), color=_color, alpha=0.3, label='__nolegend__')

    ax.set(xlim=(0.001, 20), ylim=(0, None), xscale='log', xlabel=r'$\bf D_{p}\ (nm)$',
           ylabel=r'$\bf Probability\ (dM/dlogdp)$', title=r'Log-normal Mass Size Distribution')

    ax.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax.legend()

    return ax


@set_figure
def lognorm_dist(**kwargs) -> Axes:
    """
    Plot various particle size distributions to illustrate log-normal distributions and transformations.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments.

    Examples
    --------
    Example : Plot default particle size distributions
    >>> lognorm_dist()
    """

    fig, ax = plt.subplots(2, 2, **kwargs.get('fig_kws', {}))
    ([ax1, ax2], [ax3, ax4]) = ax
    fig.suptitle('Particle Size Distribution', fontweight='bold')
    plt.subplots_adjust(left=0.125, right=0.925, bottom=0.1, top=0.93, wspace=0.4, hspace=0.4)

    # pdf
    normpdf = lambda x, mu, sigma: (1 / (sigma * sqrt(2 * pi))) * exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    lognormpdf = lambda x, gmean, gstd: (1 / (log(gstd) * sqrt(2 * pi))) * exp(
        -(log(x) - log(gmean)) ** 2 / (2 * log(gstd) ** 2))
    lognormpdf2 = lambda x, gmean, gstd: (1 / (x * log(gstd) * sqrt(2 * pi))) * exp(
        -(log(x) - log(gmean)) ** 2 / (2 * log(gstd) ** 2))

    # 生成x
    x = np.linspace(-10, 10, 1000)
    x2 = np.geomspace(0.01, 100, 1000)

    # 生成常態分布
    pdf = normpdf(x, mu=0, sigma=2)

    pdf2_1 = lognormpdf(x2, gmean=0.8, gstd=1.5)
    pdf2_2 = lognormpdf2(x2, gmean=0.8, gstd=1.5)
    ln2_1 = lognorm(scale=0.8, s=np.log(1.5))  # == lognormpdf2(x2, gmean=0.8, gstd=1.5)

    # Question 1
    # 若對數常態分布x有gmd=3, gstd=2，ln(x) ~ 常態分佈，試問其分布的平均值與標準差??
    data3 = lognorm(scale=3, s=np.log(2)).rvs(size=5000)
    Y = np.log(data3)  # Y ~ N(mu=log(gmean), sigma=log(gstd))

    # Question 2
    # 若常態分布x有平均值3 標準差1，exp(x)則為一對數常態分佈? 由對數常態分佈的定義 若隨機變數ln(Z)是常態分布 則Z為對數常態分布
    # 因此已知Z = exp(x), so ln(Z)=x，Z ~ 對數常態分佈，試問其分布的幾何平均值與幾何標準差是??
    data5 = norm(loc=3, scale=1).rvs(size=5000)
    Z = np.exp(data5)  # Z ~ LN(geoMean=exp(mu), geoStd=exp(sigma))

    def plot_distribution(ax, x, pdf, color='k-', **kwargs):
        ax.plot(x, pdf, color, **kwargs)
        ax.set_xlabel('Particle Size (micron)')
        ax.set_ylabel('Probability Density')
        ax.set_xlim(x.min(), x.max())

    # 繪製粒徑分布
    plot_distribution(ax1, x, pdf)

    plot_distribution(ax2, x2, ln2_1.pdf(x2), 'b-*')
    plot_distribution(ax2, x2, pdf2_1, 'g-')
    plot_distribution(ax2, x2, pdf2_2, 'r--')
    ax2.semilogx()

    plot_distribution(ax3, x, normpdf(x, mu=log(3), sigma=log(2)), 'k-')
    ax3.hist(Y, bins=100, density=True, alpha=0.6, color='g')

    plot_distribution(ax4, x2, lognormpdf2(x2, gmean=exp(3), gstd=exp(1)), 'r-')
    ax4.hist(Z, bins=100, density=True, alpha=0.6, color='g')
    ax4.semilogx()

    return ax


if __name__ == '__main__':
    PNSD = DataReader('PNSD_dNdlogdp.csv')
    heatmap(PNSD, unit="Number")