import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from pandas import read_csv, concat
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter
from DataPlot.templates import set_figure
from DataPlot.data_processing import main
from DataPlot.data_processing.Mie_theory import Mie_PESD


PATH_MAIN = Path(__file__).parents[1] / "Data-example"

with open(PATH_MAIN / 'Level2' / 'distribution' / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

dp = np.array(PNSD.columns, float)
dlogdp = 0.014

def Mie_SurfaceLognormal(m, wavelength, geoMean, geoStdDev, numberOfParticles, nMedium=1.0, numberOfBins=167, lower=1, upper=2500, gamma=1):
    nMedium = nMedium.real
    m /= nMedium
    wavelength /= nMedium

    ithPart = lambda gammai, dp, geoMean, geoStdDev: (
            (gammai / (np.sqrt(2 * np.pi) * np.log(geoStdDev) * dp)) * np.exp(-(np.log(dp) - np.log(geoMean)) ** 2 / (2 * np.log(geoStdDev) ** 2)))

    logdp = np.logspace(np.log10(lower), np.log10(upper), numberOfBins)

    ndp = numberOfParticles * ithPart(gamma, logdp, geoMean, geoStdDev)

    Bext, Bsca, Babs = Mie_PESD(m, wavelength, dp, logdp, ndp)

    return Bext, Bsca, Babs


def _get(reset=False):
    with open(PATH_MAIN / 'Level2' / 'distribution' / 'PESD_dextdlogdp_internal.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PESD = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level2' / 'distribution' / 'PSSD_dSdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PSSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level2' / 'distribution' / 'PVSD_dVdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PVSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    return main(reset=False)


def rsm(): # Response surface methodology (RSM)
    def function(RI, GMD):
        Z = np.zeros_like(RI)  # 使用 np.zeros_like 可以確保 Z 和 RI 具有相同的形狀

        for i in range(RI.shape[0]):
            for j in range(RI.shape[1]):
                    _RI, _mu = RI[i, j], GMD[i, j]

                    # 輸入 GMDs 和 GSDs 計算 extinction, scattering, absorption
                    Bext, Bsca, Babs = Mie_SurfaceLognormal(m=_RI, wavelength=550, geoMean=_mu, geoStdDev=2,
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
        ylabel = kwargs.get('ylabel', r'$\bf GMD (nm)$')
        zlabel = kwargs.get('zlabel', r'$\bf Extinction\ (1/Mm)$')
        title = kwargs.get('title', r'$\bf Sensitive\ test\ of\ Extinction$')
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)
        ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 1), useMathText=True)

    plot(real, gmd, ext)


def three_dimension_distribution():
    data = _get()
    line = np.arange(10)
    weighting = {'SD': {'zlim': (0, 1.5e5),
                        'label': r'$\bf dN/dlogdp\ ({\mu}m^{-1}/cm^3)$'},
                 'SSD': {'zlim': (0, 1.5e9),
                         'label': r'$\bf dS/dlogdp\ ({\mu}m/cm^3)$'},
                 'ED': {'zlim': (0, 700),
                        'label': r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'},
                 'VD': {'zlim': (0, 1e11),
                        'label': r'$\bf dV/dlogdp\ ({\mu}m^2/cm^3)$'}
                 }

    for key, subdict in weighting.items():
        # fig 1 -> dp, Ext_dis * 10
        X, Y = np.meshgrid(dp, line)
        Z = np.zeros((10, 167))
        for i in range(10): # SD SSD ED VD
            Z[i] = data[key][i]

        def log_tick_formatter(val, pos=None):
            return "{:.0f}".format(np.exp(val))
        dp_ = np.insert(dp, 0, 11.7)
        dp_extend = np.append(dp_, 2437.4)
        _X, _Y = np.meshgrid(np.log(dp_extend), line)

        _Z = np.pad(Z, ((0, 0), (1, 1)), 'constant')
        verts = []
        for i in range(_X.shape[0]):
            verts.append(list(zip(_X[i, :], _Z[i, :])))

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
        facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
        poly = PolyCollection(verts, facecolors=facecolors, edgecolors='k', lw=0.5, alpha=.7)
        ax.add_collection3d(poly, zs=range(1, 11), zdir='y')
        # ax.set_xscale('log') <- dont work
        ax.set(xlim=(np.log(50), np.log(2437.4)), ylim=(1, 10), zlim=subdict['zlim'],
               xlabel=r'$\bf D_{p}\ (nm)$', ylabel=r'$\bf $', zlabel=subdict['label'])
        ax.set_xlabel(r'$\bf D_{p}\ (nm)$', labelpad=10)
        ax.set_ylabel(r'$\bf Class$', labelpad=10)
        ax.set_zlabel(subdict['label'], labelpad=15)

        major_ticks = np.log([10, 100, 1000])
        minor_ticks = np.log([20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000])
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        ax.zaxis.get_offset_text().set_visible(False)
        exponent = int('{:.2e}'.format(np.max(Z)).split('e')[1])
        # ax.text2D(0.97, 0.80, '$\\times 10^{-7}$', transform=ax.transAxes)
        ax.text(ax.get_xlim()[1]*1.05, ax.get_ylim()[1], ax.get_zlim()[1]*1.1,
                '$\\times\\mathdefault{10^{%d}}$' % exponent)
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0), useOffset=False)
        plt.show()
    #
    #     fig.savefig(f'3D_{key}')
    pass


if __name__ == '__main__':
    rsm()



