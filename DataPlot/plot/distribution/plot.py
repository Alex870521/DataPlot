import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Optional, Literal
from scipy.stats import norm, lognorm
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.collections import PolyCollection
from DataPlot.plot.core import set_figure

color_choose = {'Clean': ['#1d4a9f', '#84a7e9'],
                'Transition': ['#4a9f1d', '#a7e984'],
                'Event': ['#9f1d4a', '#e984a7']}


@set_figure(fs=12)
def heatmap(df: pd.DataFrame,
            ax: Optional[plt.Axes] = None,
            logy: bool = True,
            cbar: bool = True,
            hide_low: bool = True,
            cmap: str = 'jet',
            fig_kws: Optional[dict] = {},
            cbar_kws: Optional[dict] = {},
            plot_kws: Optional[dict] = {},
            **kwargs):
    """ Plot the size distribution over time.

    Parameters
    ----------
    df : DataFrame
        A DataFrame of particle concentrations to plot the heatmap.
    ax : matplotlib.axis.Axis
        An axis object to plot on. If none is provided, one will be created.
    logy : bool, default=True
        If true, the y-axis will be semilogy.
    cbar : bool, default=True
        If true, a colorbar will be added.
    hide_low : bool, default=True
        If true, low values will be masked.
    cmap : matplotlib.colormap, default='viridis'
        The colormap to use. Can be anything other that 'jet'.
    fig_kws : dict, default=None
        Optional kwargs to pass to the Figure.
    cbar_kws : dict, default=None
        Optional kwargs to be passed to the colorbar.
    plot_kws : dict, default=None
        Optional kwargs to be passed to pcolormesh.

    Returns
    -------
    ax : matplotlib.axis.Axis

    Examples
    --------
    Plot a SPMS + APS data:
    >>> ax = heatmap(DataFrame, cmap='jet')
    """
    print('Plot: heatmap')

    # Copy to avoid modifying original data
    time = df.index
    dp = np.array(df.columns, dtype=float)
    data = df.copy().to_numpy()
    data = np.nan_to_num(data)

    # Set the colorbar min and max based on the min and max of the values
    cbar_min = cbar_kws.pop('cbar_min', data.min() if data.min() > 0.0 else 1.)
    cbar_max = cbar_kws.pop('cbar_max', data.max())

    # Increase values below cbar_min to cbar_min
    if hide_low:
        below_min = data < cbar_min
        data[below_min] = cbar_min

    # Set the plot_kws
    plot_kws = dict(norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max), cmap=cmap, **plot_kws)

    # Set the figure keywords
    fig_kws = dict(figsize=(10, 4), **fig_kws)

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # main plot
    pco1 = ax.pcolormesh(time, dp, data.T, shading='auto', **plot_kws)

    # Set the ylabel and ylim
    ax.set(ylabel=r'$\bf D_p\ (nm)$', ylim=(dp.min(), dp.max()))

    # Set title
    st_tm, fn_tm = time[0], time[-1]
    ax.set_title(kwargs.get('title', f'{st_tm.strftime("%Y/%m/%d")} - {fn_tm.strftime("%Y/%m/%d")}'))

    # Set the axis to be log in the y-axis
    if logy:
        ax.semilogy()
        ax.yaxis.set_major_formatter(ScalarFormatter())

    # Set the figure keywords
    if cbar:
        cbar_kws = dict(label=r'$dN/dlogD_p\ (\# / cm^{-3})$', **cbar_kws)
        clb = plt.colorbar(pco1, pad=0.01, **cbar_kws)

    # fig.savefig(f'heatmap_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')

    return ax


@set_figure
def overlay_dist(dp: np.ndarray, dist: np.ndarray, ax: Optional[plt.Axes] = None, enhancement=False, fig_kws={},
                 plot_kws={}, **kwargs):
    """
    Plot particle size distribution curves and optionally show enhancements.

    Parameters
    ----------
    dp : array_like
        Particle diameters.
    dist : dict or list
        If dict, keys are labels and values are arrays of distribution values.
        If listed, it should contain three arrays for different curves.
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot. If not provided, a new subplot will be created.
    enhancement : bool, optional
        Whether to show enhancement curves.
    fig_kws : dict, optional
        Keyword arguments for creating the figure.
    plot_kws : dict, optional
        Keyword arguments for plotting curves.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ax : AxesSubplot
        Matplotlib AxesSubplot.

    Examples
    --------
    Example 1: Plot a dictionary of distributions
    >>> overlay_dist(dp_values, {'Clean': clean_data, 'Transition': transition_data, 'Event': event_data})

    Example 2: Plot a list of distributions with custom labels
    >>> overlay_dist(dp_values, [curve1_data, curve2_data, curve3_data], labels=['Curve 1', 'Curve 2', 'Curve 3'], enhancement=True)
    """
    print('Plot: overlay_dist')

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    # plot_kws
    plot_kws = dict(ls='solid', lw=2, alpha=0.8, **plot_kws)

    # Receive input data
    if isinstance(dist, (dict, list)) and len(dist) == 3:
        if isinstance(dist, dict):
            dist = [dist['Clean'], dist['Transition'], dist['Event']]

        labels = kwargs.get('labels', ['Clean', 'Transition', 'Event'])
        colors = [color_choose[label][0] for label in labels]

        for values, label, color in zip(dist, labels, colors):
            ax.plot(dp, values, label=label, color=color, **plot_kws)

        # Area
        ax.fill_between(dp, y1=0, y2=dist[0], alpha=0.5, color=color_choose['Clean'][1])
        ax.fill_between(dp, y1=dist[0], y2=dist[1], alpha=0.5, color=color_choose['Transition'][1])
        ax.fill_between(dp, y1=dist[1], y2=dist[2], alpha=0.5, color=color_choose['Event'][1])

    else:
        raise ValueError("Invalid 'dist' format. It should be a dictionary or a list of three arrays.")

    # figure_set
    xlim = kwargs.get('xlim', (11.8, 2500))
    ylim = kwargs.get('ylim', (0, 650))
    xlabel = kwargs.get('xlabel', r'$ Diameter\ (nm)$')
    ylabel = kwargs.get('ylabel', r'$ d{\sigma}/dlogdp\ (1/Mm)$')
    title = kwargs.get('title', r'$\bf Extinction\ Distribution$')
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useMathText=True)
    ax.semilogx()

    if enhancement:
        ax2 = ax.twinx()
        enhance_1 = (dist[1] / dist[0])
        enhance_2 = (dist[2] / dist[1])
        ax2.plot(dp, (dist[1] / dist[0]), ls='dashed', color='k', lw=2, label='Enhancement ratio 1')
        ax2.plot(dp, (dist[2] / dist[1]), ls='dashed', color='gray', lw=2, label='Enhancement ratio 2')
        ax2.set(xlim=xlim, ylim=(0.5, None), xlabel=xlabel, ylabel='Enhancement ratio')

        # Combine legends from ax and ax2
        legends_combined, labels_combined = [], []
        axes_list = fig.get_axes()
        for axes in axes_list:
            legends, labels = axes.get_legend_handles_labels()
            legends_combined += legends
            labels_combined += labels

        ax.legend(legends_combined, labels_combined, loc='upper left', prop={'weight': 'bold'})

    else:
        ax.legend(loc='upper left', prop={'weight': 'bold'})

    # fig.savefig(f'multi_dist_{figname}')

    return ax


@set_figure(figsize=(10, 4))
def separate_dist(dp: np.ndarray, dist: np.ndarray, dist2: np.ndarray, dist3: np.ndarray, ax: Optional[plt.Axes] = None,
                  fig_kws={}, plot_kws={}, **kwargs):
    """
    Plot particle size distribution curves on three separate subplots.

    Parameters
    ----------
    dp : array_like
        Particle diameters.
    dist : dict
        Dictionary containing distribution data for the first subplot.
    dist2 : dict
        Dictionary containing distribution data for the second subplot.
    dist3 : dict
        Dictionary containing distribution data for the third subplot.
    ax : array_like of AxesSubplot, optional
        Matplotlib AxesSubplot array with three subplots. If not provided, a new subplot will be created.
    fig_kws : dict, optional
        Keyword arguments for creating the figure.
    plot_kws : dict, optional
        Keyword arguments for plotting curves.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ax : array_like of AxesSubplot
        Matplotlib AxesSubplot array with three subplots.

    Examples
    --------
    Example 1: Plot three distributions on separate subplots
    >>> separatedist(dp_values, {'State1': data1, 'State2': data2}, {'State1': data3, 'State2': data4}, {'State1': data5, 'State2': data6})

    Example 2: Plot with custom labels and titles
    >>> separatedist(dp_values, {'Clean': clean_data, 'Transition': transition_data}, {'Clean': clean_data2, 'Transition': transition_data2}, {'Clean': clean_data3, 'Transition': transition_data3}, labels=['Number', 'Surface', 'Volume'], title='Particle Size Distributions')
    """
    print('Plot: separatedist')

    plot_kws = dict(ls='solid', lw=2, alpha=0.8, **plot_kws)

    if ax is None:
        fig, ax = plt.subplots(1, 3, **fig_kws)
        ax1, ax2, ax3 = ax
    # ax1
    for i, state in enumerate(dist.keys()):
        ax1.plot(dp, dist[state], color=color_choose[state][0], label='__nolegend__', **plot_kws)

    # Set ax1
    xlim = kwargs.get('xlim', (11.8, 2500))
    ylim = kwargs.get('ylim', (0, 1.5e5))
    xlabel = kwargs.get('xlabel', r'$\bf Diameter\ (nm)$')
    ylabel = kwargs.get('ylabel', r'$\bf dN/dlogdp $')
    title = 'Number'
    ax1.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2), useMathText=True)
    ax1.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax1.semilogx()

    # ax2
    for i, state in enumerate(dist2.keys()):
        ax2.plot(dp, dist2[state], color=color_choose[state][0], label=f'{state}', **plot_kws)

    # Set ax2
    ylim = kwargs.get('ylim', (0, 1.5e9))
    ylabel = kwargs.get('ylabel', r'$\bf dS/dlogdp$')
    title = 'Surface'
    ax2.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2), useMathText=True)
    ax2.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax2.semilogx()
    ax2.legend(loc='upper left', prop={'weight': 'bold'})

    # ax3
    for i, state in enumerate(dist3.keys()):
        ax3.plot(dp, dist3[state], color=color_choose[state][0], label='__nolegend__', **plot_kws)

    # Set ax3
    ylim = kwargs.get('ylim', (0, 1e11))
    ylabel = kwargs.get('ylabel', r'$\bf dV/dlogdp$')
    title = 'Volume'
    ax3.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2), useMathText=True, useLocale=True)
    ax3.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax3.semilogx()

    return ax


@set_figure
def dist_with_std(dp, Ext_amb_dis, Ext_amb_dis_std, Ext_dry_dis, Ext_dry_dis_std, ax=None, fig_kws={}, **kwargs):
    """
    Plot extinction distribution with standard deviation for ambient and dry conditions.

    Parameters
    ----------
    dp : array_like
        Particle diameters.
    Ext_amb_dis : dict
        Dictionary containing extinction distribution data for ambient conditions.
    Ext_amb_dis_std : dict
        Dictionary containing standard deviation data for ambient extinction distribution.
    Ext_dry_dis : dict
        Dictionary containing extinction distribution data for dry conditions.
    Ext_dry_dis_std : dict
        Dictionary containing standard deviation data for dry extinction distribution.
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot. If not provided, a new subplot will be created.
    fig_kws : dict, optional
        Keyword arguments for creating the figure.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ax : AxesSubplot
        Matplotlib AxesSubplot.
    """
    print('Plot: dist_with_std')

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    for state in Ext_amb_dis.keys():
        PESD, PESD_std = Ext_amb_dis[state], Ext_amb_dis_std[state]
        PESD_std = np.array(pd.DataFrame(PESD_std).ewm(span=5).mean()).reshape(167, )
        PESD_low, PESD_up = PESD - PESD_std, PESD + PESD_std

        PESD_dry, PESD_std_dry = Ext_dry_dis[state], Ext_dry_dis_std[state]
        PESD_std_dry = np.array(pd.DataFrame(PESD_std_dry).ewm(span=5).mean()).reshape(167, )
        PESD_low_dry, PESD_up_dry = PESD_dry - PESD_std_dry, PESD_dry + PESD_std_dry

        ax.plot(dp, PESD, ls='solid', color=color_choose[state][0], lw=2, label=f'Amb {state}')
        ax.plot(dp, PESD_dry, ls='dashed', color=color_choose[state][1], lw=2, label=f'Dry {state}')
        ax.fill_between(dp, y1=PESD_low, y2=PESD_up, alpha=0.4, color=color_choose[state][1], edgecolor=None,
                        label='__nolegend__')
        ax.fill_between(dp, y1=PESD_low_dry, y2=PESD_up_dry, alpha=0.5, color='#ece8e7',
                        edgecolor=color_choose[state][1], label='__nolegend__')
        plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)

    # figure_set
    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, 850)
    xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'
    title = kwargs.get('title', r'$\bf Extinction\ Distribution$')
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(loc='upper left', prop={'weight': 'bold'})
    ax.semilogx()

    return ax


@set_figure
def compare(dp, dist, std1, dist2, std2, ax=None, std_scale=0.2, fig_kws={}, **kwargs):
    """
    Compare two extinction distributions and plot the percentage error.

    Parameters
    ----------
    dp : array_like
        Particle diameters.
    dist : array_like
        Extinction distribution data for the first condition.
    std1 : array_like
        Standard deviation data for the first condition.
    dist2 : array_like
        Extinction distribution data for the second condition.
    std2 : array_like
        Standard deviation data for the second condition.
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot. If not provided, a new subplot will be created.
    std_scale : float, optional
        Scaling factor for reducing the standard deviation.
    fig_kws : dict, optional
        Keyword arguments for creating the figure.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ax : AxesSubplot
        Matplotlib AxesSubplot.
    """
    print('Plot: compare')

    PESD, PESD_std = dist, std1
    PESD_std = np.array(pd.DataFrame(PESD_std).ewm(span=5).mean()).reshape(len(dist), ) * std_scale
    PESD_low, PESD_up = PESD - PESD_std, PESD + PESD_std

    PESD_dry, PESD_std_dry = dist2, std2
    PESD_std_dry = np.array(pd.DataFrame(PESD_std_dry).ewm(span=5).mean()).reshape(len(dist), ) * std_scale
    PESD_low_dry, PESD_up_dry = PESD_dry - PESD_std_dry, PESD_dry + PESD_std_dry

    # 创建两个数组
    appro = np.array(dist)
    exact = np.array(dist2)

    abs_diff = np.absolute(np.subtract(appro, exact))
    percentage_error = np.divide(abs_diff, exact) * 100
    percentage_error = np.array(pd.DataFrame(percentage_error).ewm(span=5).mean()).reshape(len(dist), )

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    ax.plot(dp, PESD, ls='solid', color=color_choose['Clean'][0], lw=2, label='Internal')
    ax.plot(dp, PESD_dry, ls='solid', color=color_choose['Transition'][0], lw=2, label='External')
    ax.fill_between(dp, y1=PESD_low, y2=PESD_up, alpha=0.3, color=color_choose['Clean'][0], edgecolor=None,
                    label='__nolegend__')
    ax.fill_between(dp, y1=PESD_low_dry, y2=PESD_up_dry, alpha=0.3, color=color_choose['Transition'][0],
                    edgecolor=None, label='__nolegend__')
    ax.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)

    # figure_set
    xlim = kwargs.get('xlim', (11.8, 2500))
    ylim = kwargs.get('ylim', (0, None))
    xlabel = kwargs.get('xlabel', r'$ Diameter\ (nm)$')
    ylabel = kwargs.get('ylabel', r'$ d{\sigma}/dlogdp\ (1/Mm)$')
    title = kwargs.get('title', r'$\bf Extinction\ Distribution$')
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.semilogx()

    ax2 = ax.twinx()
    ax2.plot(dp, percentage_error, ls='--', color='r', label='Error')
    ax2.set_ylabel('Error (%)')

    # Combine legends from ax and ax2
    legends_combined, labels_combined = [], []
    axes_list = fig.get_axes()
    for axes in axes_list:
        legends, labels = axes.get_legend_handles_labels()
        legends_combined += legends
        labels_combined += labels

    ax.legend(legends_combined, labels_combined, loc='upper left', prop={'weight': 'bold'})

    return ax


@set_figure
def ls_mode(ax=None, **kwargs):
    """
    Plot log-normal mass size distribution for small mode, large mode, and sea salt particles.

    Parameters
    ----------
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot. If not provided, a new subplot will be created.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ax : AxesSubplot
        Matplotlib AxesSubplot.

    Examples
    --------
    Example 1: Plot log-normal mass size distribution with default settings
    >>> ls_mode()

    Example 2: Plot log-normal mass size distribution with custom settings
    >>> ls_mode(ax=my_axes, xlim=(0.01, 30), ylim=(0, 0.5), xlabel='Particle Diameter (μm)',
    ...         ylabel='Probability (dM/dlogdp)', title='Custom Log-normal Mass Size Distribution')
    """
    print('Plot: ls_mode')

    if ax is None:
        fig, ax = plt.subplots()

    geoMean = [0.2, 0.5, 2.5]
    geoStdv = [2.2, 1.5, 2.0]
    color = ['g', 'r', 'b']
    label = [r'$\bf Small\ mode\ :D_{g}\ =\ 0.2\ \mu m,\ \sigma_{{g}}\ =\ 2.2$',
             r'$\bf Large\ mode\ :D_{g}\ =\ 0.5\ \mu m,\ \sigma_{{g}}\ =\ 1.5$',
             r'$\bf Sea\ salt\ :D_{g}\ =\ 2.5\ \mu m,\ \sigma_{{g}}\ =\ 2.0$',
             ]
    for _geoMean, _geoStdv, _color, _label in zip(geoMean, geoStdv, color, label):
        x = np.geomspace(0.001, 20, 10000)
        # 用logdp畫 才會讓最大值落在geoMean上
        pdf = (np.exp(-(np.log(x) - np.log(_geoMean)) ** 2 / (2 * np.log(_geoStdv) ** 2))
               / (np.log(_geoStdv) * np.sqrt(2 * np.pi)))

        ax.semilogx(x, pdf, linewidth=2, color=_color, label=_label)
        ax.fill_between(x, pdf, 0, where=(pdf > 0), interpolate=False, color=_color, alpha=0.3, label='__nolegend__')

    xlim = kwargs.get('xlim', (0.001, 20))
    ylim = kwargs.get('ylim', (0, None))
    xlabel = kwargs.get('xlabel', r'$ Diameter\ (\mu m)$')
    ylabel = kwargs.get('ylabel', r'$\bf Probability\ (dM/dlogdp)$')
    title = kwargs.get('title', r'Log-normal Mass Size Distribution')
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useMathText=True)
    ax.legend(loc='upper left', handlelength=1, frameon=False)
    ax.semilogx()

    return ax


@set_figure(fs=10, titlesize=12)
def psd_example(ax=None, **kwargs):
    """
    Plot various particle size distributions to illustrate log-normal distributions and transformations.

    Parameters
    ----------
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot. If not provided, a new subplot will be created.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ax : AxesSubplot
        Matplotlib AxesSubplot.

    Examples
    --------
    Example 1: Plot default particle size distributions
    >>> psd_example()
    """
    print('Plot: pse_example')

    if ax is None:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
    else:
        fig = ax.figure
        axes = [ax, None, None, None, None, None]

    # 给定的幾何平均粒徑和幾何平均標準差
    gmean = 1
    gstd = 1

    mu = np.log(gmean)
    sigma = np.log(gstd)

    normpdf = lambda x, mu, sigma: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    lognormpdf = lambda x, gmean, gstd: (1 / (np.log(gstd) * np.sqrt(2 * np.pi))) * np.exp(
        -(np.log(x) - np.log(gmean)) ** 2 / (2 * np.log(gstd) ** 2))
    lognormpdf2 = lambda x, gmean, gstd: (1 / (x * np.log(gstd) * np.sqrt(2 * np.pi))) * np.exp(
        -(np.log(x) - np.log(gmean)) ** 2 / (2 * np.log(gstd) ** 2))

    # 生成常態分布
    x = np.linspace(-10, 10, 1000)
    pdf = normpdf(x, mu=0, sigma=2)

    x2 = np.geomspace(0.01, 50, 1000)
    pdf2_1 = lognormpdf(x2, gmean=0.8, gstd=1.5)
    pdf2_2 = lognormpdf2(x2, gmean=0.8, gstd=1.5)

    # pdf2_2 = lognormpdf2(x2, gmean=np.exp(0.8), gstd=np.exp(1.5))
    # 這兩個相等
    ln2_1 = lognorm(s=1.5, scale=np.exp(0.8))
    ttt = lambda x, mu, std: (1 / (x * std * np.sqrt(2 * np.pi))) * np.exp(
        -(np.log(x) - np.log(mu)) ** 2 / (2 * std ** 2))

    # 若對數常態分布x有mu=3, sigma=1，ln(x)則為一常態分佈，試問其分布的平均值與標準差
    pdf3 = lognormpdf(x2, gmean=3, gstd=1.5)
    ln1 = lognorm(s=1, scale=np.exp(3))
    data3 = ln1.rvs(size=1000)

    Y = np.log(data3)  # Y.mean()=3, Y.std()=1
    nor2 = norm(loc=3, scale=1)
    data4 = nor2.rvs(size=1000)

    # 若常態分布x有平均值0.8 標準差1.5，exp(x)則為一對數常態分佈? 由對數常態分佈的定義 若隨機變數ln(Y)是常態分布 則Y為對數常態分布
    # 因此已知Y = exp(x) ln(Y)=x ~ 常態分布，Y ~ 對數常態分佈，試問其分布的平均值與標準差是?? Y ~ LN(geoMean=0.8, geoStd=1.5)
    nor3 = norm(loc=0.8, scale=1.5)
    data5 = nor3.rvs(size=1000)

    Z = np.exp(data5)
    ln3 = lognorm(s=1.5, scale=np.exp(0.8))

    data6 = ln3.rvs(size=1000)

    def plot_distribution(ax, x, pdf, color='k-', **kwargs):
        ax.plot(x, pdf, color, **kwargs)
        ax.set_title('Particle Size Distribution')
        ax.set_xlabel('Particle Size (micron)')
        ax.set_ylabel('Probability Density')

    # 繪製粒徑分布
    fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2)
    # ax1
    plot_distribution(ax1, x, pdf, linewidth=2)

    # ax2
    plot_distribution(ax2, x2, ln2_1.pdf(x2), 'b-', linewidth=2)
    plot_distribution(ax2, x2, pdf2_1, 'g-', linewidth=2)
    plot_distribution(ax2, x2, pdf2_2, 'r-', linewidth=2)
    ax2.set_xlim(0.01, 50)
    ax2.semilogx()

    # ax3
    plot_distribution(ax3, x2, pdf3, 'k-', linewidth=2)
    ax3.set_xlim(x2.min(), x2.max())
    ax3.semilogx()

    # ax4
    x = np.linspace(min(Y), max(Y), 1000)
    pdf = nor2.pdf(x)
    plot_distribution(ax4, x, pdf, 'k-', linewidth=2)

    # ax5
    x = np.linspace(min(data5), max(data5), 1000)
    plot_distribution(ax5, x, nor3.pdf(x), 'k-', linewidth=2)

    # ax6
    ax6.hist(Z, bins=5000, density=True, alpha=0.6, color='g')
    x = np.geomspace(min(Z), max(Z), 1000)
    plot_distribution(ax6, x, ln3.pdf(x), 'k-', linewidth=2)
    plot_distribution(ax6, x, lognormpdf(x, gmean=0.8, gstd=1.5), 'r-', linewidth=2)
    ax6.set_xlim(0.01, 20)
    ax6.semilogx()


@set_figure
def three_dimension(dp: np.ndarray, data: np.ndarray, weighting: Literal["PNSD", "PSSD", "PVSD", "PESD"]):
    print('Plot: three_dimension_distribution')

    mapping = {'PNSD': {'zlim': (0, 1.5e5),
                        'label': r'$\bf dN/dlogdp\ ({\mu}m^{-1}/cm^3)$'},
               'PSSD': {'zlim': (0, 1.5e9),
                        'label': r'$\bf dS/dlogdp\ ({\mu}m/cm^3)$'},
               'PESD': {'zlim': (0, 700),
                        'label': r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'},
               'PVSD': {'zlim': (0, 1e11),
                        'label': r'$\bf dV/dlogdp\ ({\mu}m^2/cm^3)$'}
               }

    def log_tick_formatter(val, pos=None):
        return "{:.0f}".format(np.exp(val))

    lines = data.shape[0]
    dp_ = np.insert(dp, 0, 11.7)
    dp_extend = np.append(dp_, 2437.4)

    _X, _Y = np.meshgrid(np.log(dp_extend), np.arange(lines))
    _Z = np.pad(data, ((0, 0), (1, 1)), 'constant')

    verts = []
    for i in range(_X.shape[0]):
        verts.append(list(zip(_X[i, :], _Z[i, :])))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
    facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
    poly = PolyCollection(verts, facecolors=facecolors, edgecolors='k', lw=0.5, alpha=.7)
    ax.add_collection3d(poly, zs=range(1, lines + 1), zdir='y')
    # ax.set_xscale('log') <- dont work
    ax.set(xlim=(np.log(50), np.log(2437.4)), ylim=(1, lines), zlim=mapping[weighting]['zlim'],
           xlabel=r'$\bf D_{p}\ (nm)$', ylabel=r'$\bf $', zlabel=mapping[weighting]['label'])
    ax.set_xlabel(r'$\bf D_{p}\ (nm)$', labelpad=10)
    ax.set_ylabel(r'$\bf Class$', labelpad=10)
    ax.set_zlabel(mapping[weighting]['label'], labelpad=15)

    major_ticks = np.log([10, 100, 1000])
    minor_ticks = np.log([20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000])
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.zaxis.get_offset_text().set_visible(False)
    exponent = int('{:.2e}'.format(np.max(data)).split('e')[1])
    ax.text(ax.get_xlim()[1] * 1.05, ax.get_ylim()[1], ax.get_zlim()[1] * 1.1,
            '$\\times\\mathdefault{10^{%d}}$' % exponent)
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0), useOffset=False)
    plt.show()
