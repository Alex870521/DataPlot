import matplotlib.pyplot as plt
import matplotlib.colors as colors
from DataPlot.plot_templates import set_figure, unit, getColor, color_maker


@set_figure(fs=12)
def heatmap(time, dp, data):
    """

    Parameters
    ----------
    time: PNSD.index
    dp: PNSD.keys().astype(float)
    data: PNSD

    Returns
    -------

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    pco1 = ax.pcolormesh(time, dp, data.interpolate(limit=2).T,
                         cmap='jet',
                         shading='auto',
                         norm=colors.PowerNorm(gamma=0.6, vmax=data.max(axis=0).quantile(0.8)))

    ax.set(yscale='log', ylim=(11.8, 1000))
    ax.set_ylabel(r'$\bf dp\ (nm)$')

    cbar = plt.colorbar(pco1, ax=ax, pad=0.01)
    cbar.set_label(r'$\bf dN/dlogdp$', labelpad=5)
    cbar.ax.ticklabel_format(axis='y', scilimits=(-2, 2), useMathText=True)
    cbar.ax.yaxis.set_offset_position('left')
    cbar.ax.yaxis.offsetText.set_fontproperties(dict(size=12))
    # fig.savefig(f'time1_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
    plt.show()
    return fig, ax
