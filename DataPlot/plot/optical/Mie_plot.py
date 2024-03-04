import math
import matplotlib.pyplot as plt
import numpy as np

from PyMieScatt import ScatteringFunction
from typing import Literal
from pathlib import Path
from DataPlot.process.method.mie_theory import Mie_Q, Mie_MEE
from DataPlot.plot import set_figure

prop_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
textprops = {'fontname': 'Times New Roman', 'weight': 'bold', 'fontsize': 16}

PATH_MAIN = Path(__file__).resolve().parent / 'Figure'

dp = np.geomspace(10, 10000, 5000)


@set_figure(figsize=(6, 6))
def Q_plot(subdic,
           x: Literal["dp", "sp"] = 'dp',
           y: Literal["Q", "MEE"] = "Q",
           **kwargs):

    dp = np.geomspace(10, 10000, 5000)

    subdic['Q'] = Mie_Q(subdic['m'], 550, dp)
    subdic['MEE'] = Mie_MEE(subdic['m'], 550, dp, subdic['density'])

    xlabel_mapping = {'dp': r'$\bf Particle\ Diameter\ (nm)$',
                      'sp': r'$\bf Size\ parameter\ (\alpha)$'}

    ylabel_mapping = {'Q': r'$\bf Optical\ efficiency\ (Q)$',
                      'MEE': r'$\bf Mass\ Optical\ Efficiency\ (m^2/g)$'}

    legend_label = {'Q': [r'$\bf Q_{{ext}}$', r'$\bf Q_{{scat}}$', r'$\bf Q_{{abs}}$'],
                    'MEE': [r'$\bf MEE$', r'$\bf MSE$', r'$\bf MAE$']}

    xlabel = xlabel_mapping.get(x, None)
    ylabel = ylabel_mapping.get(y, None)
    leg = legend_label.get(y, None)

    fig, ax = plt.subplots()

    if x == "sp":
        size_para = math.pi * dp / 550
        Q_max = subdic['Q'][0].max()
        dp_max = dp[subdic['Q'][0].argmax()]
        alp_max = size_para[subdic['Q'][0].argmax()]

        plt.vlines(x=alp_max, ymin=0, ymax=Q_max, color='gray', alpha=0.7, ls='--', lw=2.5, label='__nolegend__')
        plt.annotate(fr'$\bf \alpha\ =\ {alp_max.round(2)} = {dp_max.round(2)}\ nm $',
                     xy=(alp_max, Q_max),
                     xytext=(20, 3.5),
                     arrowprops={'color': 'blue'})
        dp = size_para

    plt.plot(dp, subdic[f'{y}'][0], color='b', label=leg[0], linestyle='-', alpha=1, lw=2.5, zorder=3)

    if x == "dp":
        plt.plot(dp, subdic[f'{y}'][1], color='g', label=leg[0], linestyle='-', alpha=1, lw=2.5)
        plt.plot(dp, subdic[f'{y}'][2], color='r', label=leg[0], linestyle='-', alpha=1, lw=2.5)
        plt.semilogx()

    plt.text(0.04, 0.92, subdic['m_format'], transform=ax.transAxes)
    plt.legend(loc='upper right', prop=prop_legend, handlelength=1.5, frameon=False)

    xlim = kwargs.get('xlim') or (dp[0], dp[-1])
    ylim = kwargs.get('ylim') or (0, None)
    xlabel = kwargs.get('xlabel') or r'$\bf Particle\ Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or ylabel
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=subdic['title'])

    plt.show()
    # fig.savefig(PATH_MAIN/f'Q_{species}')


@set_figure(figsize=(8, 6), fs=16)
def All_species_Q(dic,
                  x: Literal["dp"] = 'dp',
                  y: Literal["Q", "MEE"] = 'Q',
                  mode: Literal["ext", "sca", "abs"] = 'ext',
                  **kwargs):
    mode_mapping = {'ext': 0, 'sca': 1, 'abs': 2}

    ylabel_mapping = {'Q': {'ext': r'$\bf Extinction\ efficiency\ (Q_{{ext}})$',
                            'sca': r'$\bf Scattering\ efficiency\ (Q_{{sca}})$',
                            'abs': r'$\bf Absorption\ efficiency\ (Q_{{abs}})$'},
                      'MEE': {'ext': r'$\bf MEE\ (m^2/g)$',
                              'sca': r'$\bf MSE\ (m^2/g)$',
                              'abs': r'$\bf MAE\ (m^2/g)$'}}

    typ = mode_mapping.get(mode, None)
    ylabel = ylabel_mapping.get(y).get(mode, None)

    color = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB', '#96c8e6']
    legend_label = [fr'$\bf NH_{4}NO_{3}$', fr'$\bf (NH_{4})_{2}SO_{4}$', fr'$\bf OM$', fr'$\bf Soil$', fr'$\bf NaCl$',
                    fr'$\bf BC$', fr'$\bf Water$']

    fig, ax = plt.subplots()

    alpha = 1

    for i, (species, subdic) in enumerate(dic.items()):
        subdic['Q'] = Mie_Q(subdic['m'], 550, dp)
        subdic['MEE'] = Mie_MEE(subdic['m'], 550, dp, subdic['density'])

        plt.plot(dp, subdic[f'{y}'][typ], color=color[i], label=legend_label[i], linestyle='-', alpha=alpha, lw=2)

    plt.legend(loc='upper left', prop=prop_legend, handlelength=1.5, frameon=False)
    plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=1, alpha=0.4)
    plt.semilogx()

    xlim = kwargs.get('xlim') or (dp[0], dp[-1])
    ylim = kwargs.get('ylim')
    xlabel = kwargs.get('xlabel') or r'$\bf Particle\ Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or ylabel
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    plt.title('')
    plt.show()
    # fig.savefig(PATH_MAIN/f'Q_ALL_{mode}')


@set_figure(figsize=(12, 5))
def IJ_couple():
    """ 測試實虛部是否互相影響

    :return:
    """

    a = Mie_Q(1.50 + 0.01j, 550, dp)
    b = Mie_Q(1.50 + 0.1j, 550, dp)
    c = Mie_Q(1.50 + 0.5j, 550, dp)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    size_para = math.pi * dp / 550

    ax1.plot(size_para, a[1], 'k-', alpha=1, lw=2.5, label=r'$\bf\ k\ =\ 0.01$')
    ax1.plot(size_para, b[1], 'b-', alpha=1, lw=2.5, label=r'$\bf\ k\ =\ 0.10$')
    ax1.plot(size_para, c[1], 'g-', alpha=1, lw=2.5, label=r'$\bf\ k\ =\ 0.50$')
    ax1.legend()

    ax1.set_xlim(0, size_para[-1])
    ax1.set_ylim(0, None)
    ax1.set_xlabel(r'$\bf Size\ parameter\ (\alpha)$')
    ax1.set_ylabel(r'$\bf Scattering\ efficiency\ (Q_{{scat}})$')

    ax2.plot(size_para, a[2], 'k-', alpha=1, lw=2.5, label=r'$\bf\ k\ =\ 0.01$')
    ax2.plot(size_para, b[2], 'b-', alpha=1, lw=2.5, label=r'$\bf\ k\ =\ 0.10$')
    ax2.plot(size_para, c[2], 'g-', alpha=1, lw=2.5, label=r'$\bf\ k\ =\ 0.50$')
    ax2.legend()

    ax2.set_xlim(0, size_para[-1])
    ax2.set_ylim(0, None)
    ax2.set_xlabel(r'$\bf Size\ parameter\ (\alpha)$')
    ax2.set_ylabel(r'$\bf Absorption\ efficiency\ (Q_{{abs}})$')

    fig.suptitle(r'$\bf n\ =\ 1.50 $')
    plt.show()
    # fig.savefig(PATH_MAIN/f'IJ_couple')


@set_figure(figsize=(6, 5))
def RRI_2D(mode: Literal["ext", "sca", "abs"] = 'ext', **kwargs):
    mode_mapping = {'ext': 0, 'sca': 1, 'abs': 2}

    typ = mode_mapping.get(mode, None)

    for dp in [400, 550, 700]:
        RRI = np.linspace(1.3, 2, 100)
        IRI = np.linspace(0, 0.7, 100)
        arr = np.zeros((RRI.size, IRI.size))

        for i, I_RI in enumerate(IRI):
            for j, R_RI in enumerate(RRI):
                arr[i, j] = Mie_Q(R_RI + 1j * I_RI, 550, dp)[typ]

        fig, ax = plt.subplots(1, 1)
        plt.title(fr'$\bf dp\ = {dp}\ nm$', )
        plt.xlabel(r'$\bf Real\ part\ (n)$', )
        plt.ylabel(r'$\bf Imaginary\ part\ (k)$', )

        im = plt.imshow(arr, extent=(1.3, 2, 0, 0.7), cmap='jet', origin='lower')
        color_bar = plt.colorbar(im, extend='both')
        color_bar.set_label(label=fr'$\bf Scattering\ efficiency\ (Q_{{{mode}}})$')
        plt.show()
        # fig.savefig(PATH_MAIN/f'RRI_{mode}_{dp}')


@set_figure(figsize=(5, 5), fs=10)
def scattering_phase(m: complex = 1.55 + 0.01j,
                     wave: float = 600,
                     dp: float = 200):
    theta, _SL, _SR, _SU = ScatteringFunction(m, wave, dp)

    SL = np.append(_SL, _SL[::-1])
    SR = np.append(_SR, _SR[::-1])
    SU = np.append(_SU, _SU[::-1])

    angles = ['0', '60', '120', '180', '240', '300']

    plt.figure()
    plt.subplot(polar=True)

    theta = np.linspace(0, 2 * np.pi, len(SL))

    plt.thetagrids(range(0, 360, int(360 / len(angles))), angles)

    plt.plot(theta, SL, '-', linewidth=2, color='#115162', label='SL')
    plt.fill(theta, SL, '#afe0f5', alpha=0.5)
    plt.plot(theta, SR, '-', linewidth=2, color='#7FAE80', label='SR')
    plt.fill(theta, SR, '#b5e6c5', alpha=0.5)
    plt.plot(theta, SU, '-', linewidth=2, color='#621129', label='SU')
    plt.fill(theta, SU, '#f5afbd', alpha=0.5)

    plt.legend(prop={'weight': 'bold'}, loc='best', bbox_to_anchor=(1, 0, 0.2, 1), frameon=False)
    plt.title(r'$\bf Scattering\ phase\ function$')
    plt.show()
