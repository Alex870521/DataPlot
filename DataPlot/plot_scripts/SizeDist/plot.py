import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy import nan_to_num
from matplotlib.ticker import ScalarFormatter
from DataPlot.plot_templates import set_figure, unit, getColor, color_maker
from DataPlot.Data_processing.PSD_class import SizeDist

__all__ = ["heatmap",
           "histplot",
           "plot_NSV_dist"
           ]

color_choose = {'Clean': ['#1d4a9f', '#84a7e9'],
                'Transition': ['#4a9f1d', '#a7e984'],
                'Event': ['#9f1d4a', '#e984a7']}

dp = SizeDist().dp


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

    # Set the colorbar min and max based on the min and
    # max of the values
    cbar_min = cbar_kws.pop('cbar_min', z.min() if z.min() > 0.0 else 1.)
    cbar_max = cbar_kws.pop('cbar_max', z.max())

    if hide_low:
        # Increase values below cbar_min to cbar_min
        below_min = z < cbar_min
        z[below_min] = cbar_min

    # Set the plot_kws
    plot_kws = dict(dict(norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max), cmap=cmap), **plot_kws)

    # Set the figure keywords
    fig_kws = dict(dict(figsize=(10, 4)), **fig_kws)

    if ax is None:
        fig, ax = plt.subplots(**fig_kws)

    pco1 = ax.pcolormesh(x, y, z.T,
                         shading='auto',
                         **plot_kws)

    # Set the ylabel and ylim
    ax.set(ylabel=r'$\bf D_p\ (nm)$', ylim=(y.min(), y.max()))

    # Set title
    st_tm, fn_tm = x[0], x[-1]
    title = kwargs.get('title') or f'{st_tm.strftime("%Y/%m/%d")} - {fn_tm.strftime("%Y/%m/%d")}'
    ax.set_title(title)

    # Set the axis to be log in the y-axis
    if logy:
        ax.semilogy()
        ax.yaxis.set_major_formatter(ScalarFormatter())

    if cbar:
        # Set the figure keywords
        cbar_kws = dict(dict(label=r'$dN/dlogD_p\ (\# / cm^{-3})$'), **cbar_kws)
        clb = plt.colorbar(pco1, pad=0.01, **cbar_kws)

    # fig.savefig(f'heatmap_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')

    return ax


def histplot():
    pass


@set_figure(figsize=(10, 4), fs=12)
def plot_NSV_dist(dp, dist, dist2, dist3, ax=None, **kwargs):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ls_lst = ['dotted', 'dashed', 'solid']
    label_lst = ['Number', 'Surface', 'Volume']
    for i, state in enumerate(dist.keys()):
        a, = ax1.plot(dp, dist[state], ls='solid', color=color_choose[state][0], lw=2, alpha=0.8,
                     label='__nolegend__')

    # figure_set
    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, 1.5e5)
    xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or r'$\bf dN/dlogdp $'
    ax1.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2), useMathText=True)
    ax1.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax1.semilogx()

    # ax2
    for i, state in enumerate(dist2.keys()):
        b, = ax2.plot(dp, dist2[state], ls='solid', color=color_choose[state][0], lw=2, alpha=0.8,
                     label=f'{state}')


    ylim = kwargs.get('ylim') or (0, 1.5e9)
    ylabel = kwargs.get('ylabel') or r'$\bf dS/dlogdp$'
    ax2.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2), useMathText=True)
    ax2.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax2.semilogx()
    ax2.legend(loc='upper left', prop={'weight': 'bold'})

    # ax3
    for i, state in enumerate(dist3.keys()):
        c, = ax3.plot(dp, dist3[state], ls='solid', color=color_choose[state][0], lw=2, alpha=0.8,
                      label='__nolegend__')

    ylim = kwargs.get('ylim') or (0, 1e11)
    ylabel = kwargs.get('ylabel') or r'$\bf dV/dlogdp$'
    ax3.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2), useMathText=True, useLocale=True)
    ax3.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    ax3.semilogx()

    title = kwargs.get('title') or ''
    fig.suptitle(title)
    plt.show()
