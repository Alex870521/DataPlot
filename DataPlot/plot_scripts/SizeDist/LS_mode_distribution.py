import numpy as np
import matplotlib.pyplot as plt
from DataPlot.plot_templates import set_figure


@set_figure
def LS_mode():
    print(f'Plot: LS_mode')
    fig, ax = plt.subplots(figsize=(6, 6))
    geoMean = [0.2, 0.5, 2.5]
    geoStdv = [2.2, 1.5, 2.0]
    color = ['g', 'r', 'b']
    for _geoMean, _geoStdv, _color in zip(geoMean, geoStdv, color):
        x = np.geomspace(0.001, 20, 10000)
        # 用logdp畫 才會讓最大值落在geoMean上
        pdf = (np.exp(-(np.log(x) - np.log(_geoMean))**2 / (2 * np.log(_geoStdv)**2))
               / (np.log(_geoStdv) * np.sqrt(2 * np.pi)))

        ax.semilogx(x, pdf, linewidth=2, color=_color)
        ax.fill_between(x, pdf, 0, where=(pdf > 0), interpolate=False, color=_color, alpha=0.3, label='_nolegend_')

    plt.xlabel(r'$\bf Particle\ Diameter\ (\mu m)$')
    plt.ylabel(r'$\bf Probability\ (dM/dlogdp)$')
    plt.xlim(0.001, 20)
    plt.ylim(0, 1.5)
    plt.title(r'$\bf Lognormal\ Mass\ Size\ Distribution$')
    plt.legend([r'$\bf Small\ mode\ :D_{g}\ =\ 0.2\ \mu m,\ \sigma_{{g}}\ =\ 2.2$',
                r'$\bf Large\ mode\ :D_{g}\ =\ 0.5\ \mu m,\ \sigma_{{g}}\ =\ 1.5$',
                r'$\bf Sea\ salt\ :D_{g}\ =\ 2.5\ \mu m,\ \sigma_{{g}}\ =\ 2.0$'],
               loc='upper left', handlelength=1, frameon=False)
    plt.show()

    return fig, ax


if __name__ == '__main__':
    # LS_mode()

    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('_mpl-gallery')

    # make the data
    np.random.seed(3)
    x = 4 + np.random.normal(0, 2, 24)
    y = 4 + np.random.normal(0, 2, len(x))
    # size and color:
    sizes = np.random.uniform(15, 80, len(x))
    colors = np.random.uniform(15, 80, len(x))

    # plot
    fig, ax = plt.subplots(5, 1)

    ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()


def circle(radius=20):
    return radius * radius * np.pi

area = circle(2)