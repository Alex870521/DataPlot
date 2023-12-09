import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy import nan_to_num
from matplotlib.ticker import ScalarFormatter
from DataPlot.plot_templates import set_figure, unit, getColor, color_maker
from DataPlot.plot_scripts.SizeDist.LS_mode_distribution import ls_mode
from DataPlot.plot_scripts.SizeDist.compare_distribution import sizedist_example

__all__ = ["heatmap",
           "histplot",
           "overlay_dist",
           "separate_dist",
           "dist_with_std",
           "compare",
           "ls_mode",
           "sizedist_example"
           ]

color_choose = {'Clean': ['#1d4a9f', '#84a7e9'],
                'Transition': ['#4a9f1d', '#a7e984'],
                'Event': ['#9f1d4a', '#e984a7']}


@set_figure(fs=12)
def heatmap(x, y, z, ax=None, logy=True, cbar=True, hide_low=True,
            cmap='jet', fig_kws={}, cbar_kws={}, plot_kws={},
            **kwargs):
    """ Plot the size distribution over time.

    Parameters
    ----------
    x : array-like
        An array of times or datetime objects to plot on the x-axis.
    y : array-like
        An array of particle diameters to plot on the y-axis.
    z : 2D array-like
        A 2D-array of particle concentrations to plot on the Z axis.
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

    >>> ax = heatmap(x, y, z, cmap='jet')

    """
    # Copy to avoid modifying original data
    z = z.copy().to_numpy()
    z = nan_to_num(z)

    # Set the colorbar min and max based on the min and max of the values
    cbar_min = cbar_kws.pop('cbar_min', z.min() if z.min() > 0.0 else 1.)
    cbar_max = cbar_kws.pop('cbar_max', z.max())

    # Increase values below cbar_min to cbar_min
    if hide_low:
        below_min = z < cbar_min
        z[below_min] = cbar_min

    # Set the plot_kws
    plot_kws = dict(norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max), cmap=cmap, **plot_kws)

    # Set the figure keywords
    fig_kws = dict(figsize=(10, 4), **fig_kws)

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    pco1 = ax.pcolormesh(x, y, z.T, shading='auto', **plot_kws)

    # Set the ylabel and ylim
    ax.set(ylabel=r'$\bf D_p\ (nm)$', ylim=(y.min(), y.max()))

    # Set title
    st_tm, fn_tm = x[0], x[-1]
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


def histplot():
    pass


@set_figure
def overlay_dist(dp, dist, ax=None, enhancement=False, fig_kws={}, plot_kws={}, **kwargs):
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
def separate_dist(dp, dist, dist2, dist3, ax=None, fig_kws={}, plot_kws={}, **kwargs):
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

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    for state in Ext_amb_dis.keys():
        PESD, PESD_std= Ext_amb_dis[state], Ext_amb_dis_std[state]
        PESD_std = np.array(pd.DataFrame(PESD_std).ewm(span=5).mean()).reshape(167,)
        PESD_low, PESD_up = PESD - PESD_std, PESD + PESD_std

        PESD_dry, PESD_std_dry= Ext_dry_dis[state], Ext_dry_dis_std[state]
        PESD_std_dry = np.array(pd.DataFrame(PESD_std_dry).ewm(span=5).mean()).reshape(167,)
        PESD_low_dry, PESD_up_dry = PESD_dry - PESD_std_dry, PESD_dry + PESD_std_dry

        ax.plot(dp, PESD, ls='solid', color=color_choose[state][0], lw=2, label=f'Amb {state}')
        ax.plot(dp, PESD_dry, ls='dashed', color=color_choose[state][1], lw=2, label=f'Dry {state}')
        ax.fill_between(dp, y1=PESD_low, y2=PESD_up, alpha=0.4, color=color_choose[state][1], edgecolor=None, label='__nolegend__')
        ax.fill_between(dp, y1=PESD_low_dry, y2=PESD_up_dry, alpha=0.5, color='#ece8e7', edgecolor=color_choose[state][1], label='__nolegend__')
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
    ax.fill_between(dp, y1=PESD_low, y2=PESD_up, alpha=0.3, color=color_choose['Clean'][0], edgecolor=None, label='__nolegend__')
    ax.fill_between(dp, y1=PESD_low_dry, y2=PESD_up_dry, alpha=0.3, color=color_choose['Transition'][0], edgecolor=None, label='__nolegend__')
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
def ls_mode(ax=None, fig_kws={}, **kwargs):
    print(f'Plot: LS_mode')
    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

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
        pdf = (np.exp(-(np.log(x) - np.log(_geoMean))**2 / (2 * np.log(_geoStdv)**2))
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