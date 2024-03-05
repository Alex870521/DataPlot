import numpy as np
import matplotlib.pyplot as plt
from DataPlot.plot import set_figure
from DataPlot.process.method.mie_theory import Mie_Lognormal


def rsm():
    print('Plot: Response surface methodology (RSM)')

    def function(RI, GMD):
        Z = np.zeros_like(RI)  # 使用 np.zeros_like 可以確保 Z 和 RI 具有相同的形狀

        for i in range(RI.shape[0]):
            for j in range(RI.shape[1]):
                _RI, _GMD = RI[i, j], GMD[i, j]

                Bext, Bsca, Babs = Mie_Lognormal(m=_RI, wavelength=550, geoMean=_GMD, geoStdDev=2,
                                                 numberOfParticles=5e6)
                Z[i, j] = np.sum(Bext)

        return Z

    # 假設 RI、GSD、GMD
    RI = np.linspace(1.33, 1.6, 50)
    GMD = np.linspace(60, 400, 50)

    # 建立三維 meshgrid
    real, gmd = np.meshgrid(RI, GMD, indexing='xy')

    # Result
    ext = function(real, gmd)

    @set_figure
    def plot(x, y, z, **kwargs):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'), edgecolor='none')

        xlabel = kwargs.get('xlabel', r'$\bf Real\ part\ (n)$')
        ylabel = kwargs.get('ylabel', r'$\bf GMD\ (nm)$')
        zlabel = kwargs.get('zlabel', r'$\bf Extinction\ (1/Mm)$')
        title = kwargs.get('title', r'$\bf Sensitive\ test\ of\ Extinction$')
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)

        ax.zaxis.get_offset_text().set_visible(False)
        exponent = int('{:.2e}'.format(np.max(z)).split('e')[1])
        ax.text(ax.get_xlim()[1] * 1.01, ax.get_ylim()[1], ax.get_zlim()[1] * 1.1,
                '$\\times\\mathdefault{10^{%d}}$' % exponent)
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0), useOffset=False)

    plot(real, gmd, ext)


if __name__ == '__main__':
    rsm()

