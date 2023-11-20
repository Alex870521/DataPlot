import numpy as np
import matplotlib.pyplot as plt
from template import set_figure, unit, getColor, prop_legend


@set_figure
def LS_mode():
    print(f'Plot: LS_mode')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, constrained_layout=True)
    geoMean = [0.2, 0.5, 2.5]
    geoStdv = [2.2, 1.5, 2.0]
    color = ['g', 'r', 'b']
    for _geoMean, _geoStdv, _color in zip(geoMean, geoStdv, color):
        x = np.geomspace(0.001, 20, 10000)
        y = np.log(x)
        # 用logdp畫 才會讓最大值落在geoMean上
        pdf = (np.exp(-(y - np.log(_geoMean))**2 / (2 * np.log(_geoStdv)**2))
               / (np.log(_geoStdv) * np.sqrt(2 * np.pi)))


        ax.semilogx(x, pdf, linewidth=2, color=_color)
        plt.xlabel(r'$\bf Particle\ Diameter\ (\mu m)$')
        plt.ylabel(r'$\bf Probability\ (dM/dlogdp)$')
        plt.xlim(0.001, 20)
        plt.ylim(0, 1.5)
        plt.title(r'$\bf Lognormal\ Mass\ Size\ Distribution$')
        plt.legend([r'$\bf Small\ mode\ :D_{g}\ =\ 0.2\ \mu m,\ \sigma_{{g}}\ =\ 2.2$',
                    r'$\bf Large\ mode\ :D_{g}\ =\ 0.5\ \mu m,\ \sigma_{{g}}\ =\ 1.5$',
                    r'$\bf Sea\ salt\ :D_{g}\ =\ 2.5\ \mu m,\ \sigma_{{g}}\ =\ 2.0$'],
                   loc='upper left', prop=prop_legend, handlelength=1, frameon=False)
        plt.show()
        # fig.savefig('LS_mode')

    return fig, ax


if __name__ == '__main__':
    LS_mode()
