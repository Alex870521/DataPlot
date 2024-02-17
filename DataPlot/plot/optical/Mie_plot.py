import math
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from DataPlot.process.method.mie_theory import Mie_Q
from DataPlot.plot import set_figure

prop_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
textprops = {'fontname': 'Times New Roman', 'weight': 'bold', 'fontsize': 16}

dp = np.geomspace(10, 10000, 5000)

PATH_MAIN = Path(__file__).resolve().parent / 'Figure'


@set_figure(figsize=(6, 6))
def Q_plot(species, subdic):
    fig, ax = plt.subplots(1, 1)

    plt.plot(dp, subdic['Q'][0], color='b', linestyle='-', alpha=1, lw=2.5, zorder=3)
    plt.plot(dp, subdic['Q'][1], color='g', linestyle='-', alpha=1, lw=2.5)
    plt.plot(dp, subdic['Q'][2], color='r', linestyle='-', alpha=1, lw=2.5)
    plt.text(0.04, 0.92, subdic['m_format'], transform=ax.transAxes)
    plt.semilogx()
    plt.axis([dp[0], dp[-1], 0, 5])
    plt.legend([r'$\bf Q_{{ext}}$', r'$\bf Q_{{scat}}$', r'$\bf Q_{{abs}}$'], loc='upper right', prop=prop_legend,
               handlelength=1.5, frameon=False)
    plt.xlabel(r'$\bf Particle\ Diameter\ (nm)$')
    plt.ylabel(r'$\bf Optical\ efficiency\ (Q)$')
    plt.title(subdic['title'])
    plt.show()
    fig.savefig(PATH_MAIN/f'Q_{species}')


@set_figure(figsize=(6, 6))
def MEE_plot(species, subdic):
    fig, ax = plt.subplots(1, 1)

    plt.plot(dp, subdic['MEE'][0], 'b-', alpha=0.7, lw=2.5)
    plt.plot(dp, subdic['MEE'][1], 'g-', alpha=0.7, lw=2.5)
    plt.plot(dp, subdic['MEE'][2], 'r-', alpha=0.7, lw=2.5)
    plt.semilogx()
    plt.axis([dp[0], dp[-1], 0, 15])
    plt.legend([r'$\bf MEE$', r'$\bf MSE$', r'$\bf MAE$'], prop=prop_legend, frameon=False)
    plt.xlabel(r'$\bf Particle\ Diameter\ (nm)$')
    plt.ylabel(r'$\bf Mass\ Optical\ Efficiency\ (m^2/g)$')
    plt.title(subdic['title'])
    plt.show()
    fig.savefig(PATH_MAIN/f'MEE_{species}')


@set_figure(figsize=(6, 6))
def Q_size_para_plot(species, subdic):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150, constrained_layout=True)
    size_para = math.pi * dp / 550
    Q_max = subdic['Q'][0].max()
    dp_max = dp[subdic['Q'][0].argmax()]
    alp_max = size_para[subdic['Q'][0].argmax()]

    # plt.hlines(y=2, xmin=0, xmax=size_para[-1], color='gray', alpha=0.7, ls='--', lw=2.5)
    plt.vlines(x=alp_max, ymin=0, ymax=Q_max, color='gray', alpha=0.7, ls='--', lw=2.5)
    plt.plot(size_para, subdic['Q'][0], 'k-', alpha=1, label='Ext efficiency', lw=2.5)
    plt.annotate(fr'$\bf \alpha\ =\ {alp_max.round(2)} = {dp_max.round(2)}\ nm $', xy=(alp_max, Q_max),
                 xytext=(13, 4.5),
                 arrowprops={'color': 'blue'})
    plt.xlabel(r'$\bf Size\ parameter\ (\alpha)$')
    plt.ylabel(r'$\bf Extinction\ efficiency\ (Q_{{ext}})$')
    plt.xlim(0, size_para[-500])
    plt.ylim(0, 5)
    plt.title(subdic['title'])
    plt.show()
    fig.savefig(PATH_MAIN/f'Q_sp_{species}')


@set_figure(figsize=(8, 6), fs=16)
def All_species_Q(dic, x='dp', y='Q', mode='ext', **kwargs):
    """

    :param dic:
    :param x:
    :param y:
    :param mode: 'ext', 'scat', 'abs'
    :param kwargs:
    :return:
    """
    fig, ax = plt.subplots(1, 1)
    if mode == 'ext':
        type = 0
        label = r'$\bf Extinction\ efficiency\ (Q_{{ext}})$'
    if mode == 'sca':
        type = 1
        label = r'$\bf Scattering\ efficiency\ (Q_{{sca}})$'
    if mode == 'abs':
        type = 2
        label = r'$\bf Absorption\ efficiency\ (Q_{{abs}})$'

    color = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB', '#96c8e6']
    # legend_label = [fr'$ NH_{4}NO_{3}$', fr'$ (NH_{4})_{2}SO_{4}$', fr'$ OM$', fr'$ Soil$', fr'$ NaCl$',
    #                 fr'$ BC$', fr'$ Water$']

    legend_label = [fr'$\bf NH_{4}NO_{3}$', fr'$\bf (NH_{4})_{2}SO_{4}$', fr'$\bf OM$', fr'$\bf Soil$', fr'$\bf NaCl$',
                    fr'$\bf BC$', fr'$\bf Water$']

    if x == 'dp':
        alpha = 1
        for i, (species, _dic) in enumerate(dic.items()):
            plt.plot(dp, _dic['Q'][type], color=color[i], linestyle='-', alpha=alpha, lw=2)

        plt.legend(loc='upper left', labels=legend_label, prop=prop_legend, handlelength=1.5, frameon=False, )
        plt.semilogx()
        plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=1, alpha=0.4)

        xlim = kwargs.get('xlim') or (dp[0], dp[-1])
        ylim = kwargs.get('ylim') or (0, 5)
        xlabel = kwargs.get('xlabel') or r'$\bf Particle\ Diameter\ (nm)$'
        ylabel = kwargs.get('ylabel') or label
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

        plt.title('')
        plt.show()
        fig.savefig(PATH_MAIN/f'Q_ALL_{mode}', transparent=True)


@set_figure(figsize=(8, 6), fs=16)
def All_species_MEE(dic, x='dp', y='MEE', mode='ext', **kwargs):
    """

    :param dic:
    :param x:
    :param y:
    :param mode: 'ext', 'scat', 'abs'
    :param kwargs:
    :return:
    """
    fig, ax = plt.subplots(1, 1)
    if mode == 'ext':
        type = 0
        ylabel = r'$\bf MEE\ (m^2/g)$'
    if mode == 'sca':
        type = 1
        ylabel = r'$\bf MSE\ (m^2/g)$'
    if mode == 'abs':
        type = 2
        ylabel = r'$\bf MAE\ (m^2/g)$'

    color = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB', '#96c8e6']
    legend_label = [fr'$\bf NH_{4}NO_{3}$', fr'$\bf (NH_{4})_{2}SO_{4}$', fr'$\bf OM$', fr'$\bf Soil$', fr'$\bf NaCl$',
                    fr'$\bf BC$', fr'$\bf Water$']

    if x == 'dp':
        alpha = 1
        for i, (species, _dic) in enumerate(dic.items()):
            plt.plot(dp, _dic['MEE'][type], color=color[i], linestyle='-', alpha=alpha, label=legend_label[i], lw=2)

        plt.legend(loc='upper left', labels=legend_label, prop=prop_legend, handlelength=1.5, frameon=False)
        plt.semilogx()
        plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=1, alpha=0.4)

        xlim = kwargs.get('xlim') or (dp[0], dp[-1])
        ylim = kwargs.get('ylim') or (0, 11)
        xlabel = kwargs.get('xlabel') or r'$\bf Particle\ Diameter\ (nm)$'
        ylabel = kwargs.get('ylabel') or ylabel
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        plt.title('')
        plt.show()
        fig.savefig(PATH_MAIN/f'MEE_ALL_{mode}', transparent=True)


@set_figure(figsize=(12, 6))
def IJ_couple():
    """ 測試實虛部是否互相影響

    :return:
    """

    a = Mie_Q(1.50 + 0.01j, 550, dp)
    b = Mie_Q(1.50 + 0.1j, 550, dp)
    c = Mie_Q(1.50 + 0.5j, 550, dp)
    fig, (axes1, axes2) = plt.subplots(1, 2)
    size_para = math.pi * dp / 550

    abs_line1, = axes1.plot(size_para, a[2], 'k-', alpha=1, lw=2.5)
    abs_line2, = axes1.plot(size_para, b[2], 'b-', alpha=1, lw=2.5)
    abs_line3, = axes1.plot(size_para, c[2], 'g-', alpha=1, lw=2.5)
    axes1.legend(handles=[abs_line1, abs_line2, abs_line3],
                 labels=[r'$\bf\ k\ =\ 0.01$', r'$\bf\ k\ =\ 0.1$', r'$\bf\ k\ =\ 0.50$'], handlelength=1,
                 frameon=False)

    axes1.set_xlim(0, size_para[-1])
    axes1.set_ylim(0, 2)
    axes1.set_xlabel(r'$\bf Size\ parameter\ (\alpha)$')
    axes1.set_ylabel(r'$\bf Absorption\ efficiency\ (Q_{{abs}})$')

    sca_line1, = axes2.plot(size_para, a[1], 'k-', alpha=1, lw=2.5)
    sca_line2, = axes2.plot(size_para, b[1], 'b-', alpha=1, lw=2.5)
    sca_line3, = axes2.plot(size_para, c[1], 'g-', alpha=1, lw=2.5)
    axes2.legend(handles=[sca_line1, sca_line2, sca_line3],
                 labels=[r'$\bf\ k\ =\ 0.01$', r'$\bf\ k\ =\ 0.1$', r'$\bf\ k\ =\ 0.50$'], handlelength=1,
                 frameon=False)
    plt.xlim(0, size_para[-1])
    plt.ylim(0, 5)
    plt.xlabel(r'$\bf Size\ parameter\ (\alpha)$')
    plt.ylabel(r'$\bf Scattering\ efficiency\ (Q_{{scat}})$')

    plt.title(r'$\bf n\ =\ 1.50 $')
    plt.show()
    fig.savefig(PATH_MAIN/f'IJ_couple')


@set_figure(figsize=(6, 5))
def RRI_2D(mode='ext', **kwargs):
    """

    :param mode: 'ext', 'scat', 'abs'
    :return:
    """
    if mode == 'ext':
        type = 0

    if mode == 'sca':
        type = 1

    if mode == 'abs':
        type = 2

    for dp in [400, 550, 700]:
        RRI = np.linspace(1.3, 2, 100)
        IRI = np.linspace(0, 0.7, 100)
        arr = np.zeros((RRI.size, IRI.size))

        for i, I_RI in enumerate(IRI):
            for j, R_RI in enumerate(RRI):
                arr[i, j] = Mie_Q(R_RI + 1j * I_RI, 550, dp)[type]

        fig, ax = plt.subplots(1, 1)
        plt.title(fr'$\bf dp\ = {dp}\ nm$', )
        plt.xlabel(r'$\bf Real\ part\ (n)$', )
        plt.ylabel(r'$\bf Imaginary\ part\ (k)$', )

        im = plt.imshow(arr, extent=[1.3, 2, 0, 0.7], cmap='jet', origin='lower')
        color_bar = plt.colorbar(im, extend='both')
        color_bar.set_label(label=fr'$\bf Scattering\ efficiency\ (Q_{{{mode}}})$')
        plt.show()
        fig.savefig(PATH_MAIN/f'RRI_{mode}_{dp}')