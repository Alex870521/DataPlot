import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import PyMieScatt.Mie as PyMie
import matplotlib.ticker as mticker

from pandas import read_csv, concat
from matplotlib.collections import PolyCollection
from DataPlot.Data_processing import main
from pathlib import Path
from DataPlot.Data_processing.Mie_plus import Mie_PESD
from functools import lru_cache

PATH_MAIN = Path("C:/Users/alex/PycharmProjects/DataPlot/Data")

with open(PATH_MAIN / 'Level2' / 'distribution' / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

dp = np.array(PNSD.columns, dtype='float')
_length = np.size(dp)
dlogdp = [0.014] * _length


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
    if ((PATH_MAIN / '3D_plot.pkl').exists()) & (~reset):
        with open(PATH_MAIN / '3D_plot.pkl', 'rb') as f:
            return pickle.load(f)

    else:
        with open(PATH_MAIN / 'Level2' / 'distribution' / 'PESD_dextdlogdp_internal.csv', 'r', encoding='utf-8', errors='ignore') as f:
            PESD = read_csv(f, parse_dates=['Time']).set_index('Time')

        with open(PATH_MAIN / 'Level2' / 'distribution' / 'PSSD_dSdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
            PSSD = read_csv(f, parse_dates=['Time']).set_index('Time')

        with open(PATH_MAIN / 'Level2' / 'distribution' / 'PVSD_dVdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
            PVSD = read_csv(f, parse_dates=['Time']).set_index('Time')

        df = main(reset=False)

        x_bin = [f'E_{x}' for x in np.arange(1, 11) * 10]

        df['x'] = pd.qcut(df['Extinction'], 10, labels=x_bin)
        new_df = concat([df[['x', 'n_amb', 'k_amb', 'GMDn', 'GMDs', 'GMDv', 'GSDn', 'GSDs', 'GSDv']], PNSD, PESD, PSSD, PVSD],
                        axis=1).dropna()

        df_x_group = new_df.groupby('x')
        data = {'n_amb': [],
                'k_amb': [],
                'GMDn': [],
                'GSDn': [],
                'GMDs': [],
                'GSDs': [],
                'GMDv': [],
                'GSDv': [],
                'SD': [],
                'SSD':[],
                'ED': [],
                'VD':[]
                }

        for _grp, _df in df_x_group:
            Mean = _df.mean()
            for key, lst_contain in data.items():
                if key == 'SD':
                    lst_contain.append(Mean[8:175].values)
                elif key == 'ED':
                    lst_contain.append(Mean[175:342].values)
                elif key == 'SSD':
                    lst_contain.append(Mean[342:509].values)
                elif key == 'VD':
                    lst_contain.append(Mean[509:].values)
                else:
                    lst_contain.append(Mean[key])

        with open(PATH_MAIN / '3D_plot.pkl', 'wb') as f:
            pickle.dump(data, f)
        return data


if __name__ == '__main__':
    data = _get()
    RI = np.linspace(1.33, 1.6, 50)
    GMD = np.linspace(60, 400, 50)
    GSD = np.linspace(4.0, 1.9, 50)
    line = np.arange(10)

    @lru_cache(maxsize=None)
    def function(RI, GMD, GSD):
        Z = np.zeros(shape=RI.shape)
        for i, (lst_RI, lst_mu, lst_sigma) in enumerate(zip(RI, GMD, GSD)):
            for j, (_RI, _mu, _sigma) in enumerate(zip(lst_RI, lst_mu, lst_sigma)):
                # 輸入GMDs GSDs 計算extinction, scattering, absorption
                Bext, Bsca, Babs = Mie_SurfaceLognormal(m=_RI, wavelength=550, geoMean=_mu, geoStdDev=_sigma, numberOfParticles=5e6)
                Z[i, j] = np.sum(Bext)
        return Z


    X, Y = np.meshgrid(RI, GMD)
    Z, Y = np.meshgrid(GSD, GMD)
    Z = function(X, Y, Z)

    # fig 2 -> RI, GMD, Ext
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'), edgecolor='none')

    ax.set_title(r'$\bf Sensitive\ test\ of\ Extinction$')
    ax.set_xlabel(r'$\bf Real\ part\ (n)$', labelpad=10)
    ax.set_ylabel(r'$\bf GMD_s (nm)$', labelpad=10)
    ax.set_zlabel(r'$\bf Ext\ (1/Mm)$', labelpad=10)
    plt.show()

    # weighting = {'SD': {'zlim': (0, 1.5e5),
    #                     'label': r'$\bf dN/dlogdp\ ({\mu}m^{-1}/cm^3)$'},
    #              'SSD': {'zlim': (0, 1.5e9),
    #                      'label': r'$\bf dS/dlogdp\ ({\mu}m/cm^3)$'},
    #              'ED': {'zlim': (0, 700),
    #                     'label': r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'},
    #              'VD': {'zlim': (0, 1e11),
    #                     'label': r'$\bf dV/dlogdp\ ({\mu}m^2/cm^3)$'}
    #              }
    #
    # for key, subdict in weighting.items():
    #     # fig 1 -> dp, Ext_dis * 10
    #     X, Y = np.meshgrid(dp, line)
    #     Z = np.zeros((10, 167))
    #     for i in range(10): # SD SSD ED VD
    #         Z[i] = data[key][i]
    #
    #     def log_tick_formatter(val, pos=None):
    #         return "{:.0f}".format(np.exp(val))
    #     dp_ = np.insert(dp, 0, 11.7)
    #     dp_extend = np.append(dp_, 2437.4)
    #     _X, _Y = np.meshgrid(np.log(dp_extend), line)
    #
    #     _Z = np.pad(Z, ((0, 0), (1, 1)), 'constant')
    #     verts = []
    #     for i in range(_X.shape[0]):
    #         verts.append(list(zip(_X[i, :], _Z[i, :])))
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = plt.axes(projection='3d')
    #     facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
    #     poly = PolyCollection(verts, facecolors=facecolors, edgecolors='k', lw=0.5, alpha=.7)
    #     ax.add_collection3d(poly, zs=range(1, 11), zdir='y')
    #     # ax.set_xscale('log') <- dont work
    #     ax.set(xlim=(np.log(50), np.log(2437.4)), ylim=(1, 10), zlim=subdict['zlim'],
    #            xlabel=r'$\bf D_{p}\ (nm)$', ylabel=r'$\bf $', zlabel=subdict['label'])
    #     ax.set_xlabel(r'$\bf D_{p}\ (nm)$', labelpad=10)
    #     ax.set_ylabel(r'$\bf Class$', labelpad=10)
    #     ax.set_zlabel(subdict['label'], labelpad=15)
    #
    #     major_ticks = np.log([10, 100, 1000])
    #     minor_ticks = np.log([20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000])
    #     ax.set_xticks(major_ticks)
    #     ax.set_xticks(minor_ticks, minor=True)
    #     ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    #     ax.zaxis.get_offset_text().set_visible(False)
    #     exponent = int('{:.2e}'.format(np.max(Z)).split('e')[1])
    #     # ax.text2D(0.97, 0.80, '$\\times 10^{-7}$', transform=ax.transAxes)
    #     ax.text(ax.get_xlim()[1]*1.05, ax.get_ylim()[1], ax.get_zlim()[1]*1.1,
    #             '$\\times\\mathdefault{10^{%d}}$' % exponent)
    #     ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0), useOffset=False)
    #     plt.show()
    #
    #     fig.savefig(f'3D_{key}')


