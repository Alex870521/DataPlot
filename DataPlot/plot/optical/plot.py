import math
import matplotlib.pyplot as plt
import numpy as np

from PyMieScatt import ScatteringFunction
from typing import Literal, Sequence, Union, List
from pathlib import Path
from DataPlot.process.method.mie_theory import Mie_Q, Mie_MEE
from DataPlot.plot import set_figure

PATH_MAIN = Path(__file__).parent / 'Figure'

dp = np.geomspace(10, 10000, 5000)

RI_dic = {'AS': 1.53 + 0j,
          'AN': 1.55 + 0j,
          'OM': 1.54 + 0j,
          'Soil': 1.56 + 0.01j,
          'SS': 1.54 + 0j,
          'BC': 1.80 + 0.54j,
          'Water': 1.333 + 0j, }

Density_dic = {'AS': 1.73,
               'AN': 1.77,
               'OM': 1.40,
               'Soil': 2.60,
               'SS': 1.90,
               'BC': 1.50,
               'Water': 1}

legend_mapping = {'AS': fr'$\bf NH_{4}NO_{3}$',
                  'AN': fr'$\bf (NH_{4})_{2}SO_{4}$',
                  'OM': fr'$\bf OM$',
                  'Soil': fr'$\bf Soil$',
                  'SS': fr'$\bf NaCl$',
                  'BC': fr'$\bf BC$',
                  'Water': fr'$\bf Water$'}

color_mapping = {'AS': '#A65E58',
                 'AN': '#A5BF6B',
                 'OM': '#F2BF5E',
                 'Soil': '#3F83BF',
                 'SS': '#B777C2',
                 'BC': '#D1CFCB',
                 'Water': '#96c8e6'}

combined_dict = {key: {'m': value,
                       'density': Density_dic[key]} for key, value in RI_dic.items()}


@set_figure(figsize=(6, 6))
def Q_plot(
        species: Union[Literal["AS", "AN", "OM", "Soil", "SS", "BC", "Water"], List[Literal["AS", "AN", "OM", "Soil", "SS", "BC", "Water"]]],
        x: Literal["dp", "sp"] = 'dp',
        y: Literal["Q", "MEE"] = "Q",
        mode: Literal["ext", "sca", "abs"] = 'ext',
        **kwargs):
    dp = np.geomspace(10, 10000, 5000)

    mode_mapping = {'ext': 0, 'sca': 1, 'abs': 2}

    xlabel_mapping = {'dp': r'$\bf Particle\ Diameter\ (nm)$',
                      'sp': r'$\bf Size\ parameter\ (\alpha)$'}

    ylabel_mapping = {'Q': {'ext': r'$\bf Extinction\ efficiency\ (Q_{{ext}})$',
                            'sca': r'$\bf Scattering\ efficiency\ (Q_{{sca}})$',
                            'abs': r'$\bf Absorption\ efficiency\ (Q_{{abs}})$'},
                      'MEE': {'ext': r'$\bf MEE\ (m^2/g)$',
                              'sca': r'$\bf MSE\ (m^2/g)$',
                              'abs': r'$\bf MAE\ (m^2/g)$'}}

    typ = mode_mapping.get(mode, None)
    xlabel = xlabel_mapping.get(x, None)
    ylabel = ylabel_mapping.get(y, None).get(mode, None)

    fig, ax = plt.subplots()

    if x == "sp":
        size_para = math.pi * dp.copy() / 550
        dp_ = size_para

    if x == "dp":
        plt.semilogx()
        dp_ = dp.copy()

    if isinstance(species, list):
        for i, specie in enumerate(species):
            subdic = combined_dict[specie]
            leg = legend_mapping.get(specie, None)
            color = color_mapping.get(specie, None)

            subdic['Q'] = Mie_Q(subdic['m'], 550, dp)
            subdic['MEE'] = Mie_MEE(subdic['m'], 550, dp, subdic['density'])

            plt.plot(dp_, subdic[f'{y}'][typ], color=color, label=leg, linestyle='-', alpha=1, lw=2.5, zorder=3)

    else:
        legend_label = {'Q': [r'$\bf Q_{{ext}}$', r'$\bf Q_{{scat}}$', r'$\bf Q_{{abs}}$'],
                        'MEE': [r'$\bf MEE$', r'$\bf MSE$', r'$\bf MAE$']}

        ylabel_mapping = {'Q': r'$\bf Optical\ efficiency\ (Q)$',
                          'MEE': r'$\bf Mass\ Optical\ Efficiency\ (m^2/g)$'}

        subdic = combined_dict[species]
        leg = legend_label.get(y, None)
        ylabel = ylabel_mapping.get(y, None)

        subdic['Q'] = Mie_Q(subdic['m'], 550, dp)
        subdic['MEE'] = Mie_MEE(subdic['m'], 550, dp, subdic['density'])

        plt.plot(dp_, subdic[f'{y}'][0], color='b', label=leg[0], linestyle='-', alpha=1, lw=2.5, zorder=3)
        plt.plot(dp_, subdic[f'{y}'][1], color='g', label=leg[1], linestyle='-', alpha=1, lw=2.5)
        plt.plot(dp_, subdic[f'{y}'][2], color='r', label=leg[2], linestyle='-', alpha=1, lw=2.5)
        plt.text(0.04, 0.92, legend_mapping[species], transform=ax.transAxes)

    plt.legend(loc='best', prop={'weight': 'normal', 'size': 14}, handlelength=1.5, frameon=False)
    plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=1, alpha=0.4)

    xlim = kwargs.get('xlim') or (dp_[0], dp_[-1])
    ylim = kwargs.get('ylim') or (0, None)
    xlabel = kwargs.get('xlabel') or xlabel
    ylabel = kwargs.get('ylabel') or ylabel
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    plt.show()
    # fig.savefig(PATH_MAIN/f'Q_{species}')


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
