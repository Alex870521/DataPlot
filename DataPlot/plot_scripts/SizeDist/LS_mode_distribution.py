import numpy as np
import matplotlib.pyplot as plt
from DataPlot.plot_templates import set_figure


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
