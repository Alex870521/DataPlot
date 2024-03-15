import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DataPlot.process import *
from DataPlot.plot.core import *
from DataPlot.plot.templates import *

df = DataBase
dic_grp_sea = DataClassifier(df, 'Season', statistic='Dict')
dic_grp_sta = DataClassifier(df, 'State', statistic='Dict')

bins = np.linspace(0, 400, 11)
dic_grp_Ext = DataClassifier(df, by='Extinction', cut=True, statistic='Dict',
                             bins=bins,
                             labels=(bins + (bins[1] - bins[0]) / 2)[:-1])

dic_grp_RH = DataClassifier(df, by='RH', cut=True, statistic='Dict',
                            bins=np.array([0, 40, 60, 80, 100]),
                            labels=['0-40', '40~60', '60~80', '80~100'])

Species1 = ['AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry']
Species2 = ['AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry', 'ALWC_ext']
Species3 = ['AS_ext', 'AN_ext', 'OM_ext', 'Soil_ext', 'SS_ext', 'EC_ext']

water = ['ALWC_ext']
water2 = ['ALWC_AS_ext', 'ALWC_AN_ext', 'ALWC_SS_ext']

mass_1 = ['AS', 'AN', 'OM', 'Soil', 'SS', 'EC']
items = ['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'gRH']
mass_3 = ['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'ALWC']
mass_4 = ['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'ALWC']

States1 = ['Clean', 'Transition', 'Event']

mass_comp1_dict = {state: [dic_grp_sta[state][specie].mean() for specie in mass_1] for state in States1}
mass_comp3_dict = {state: [dic_grp_sta[state][specie].mean() for specie in mass_3] for state in States1}
mass_comp4_dict = {state: [dic_grp_sta[state][specie].mean() for specie in mass_4] for state in States1}
mass_comp4_dict_std = {state: [dic_grp_sta[state][specie].std() for specie in mass_4] for state in States1}

ext_dry_dict = {state: [dic_grp_sta[state][specie].mean() for specie in Species1] for state in States1}
ext_dry_std = {state: [dic_grp_sta[state][specie].std() for specie in Species1] for state in States1}
ext_amb_dict = {state: [dic_grp_sta[state][specie].mean() for specie in Species2] for state in States1}
ext_amb_std = {state: [dic_grp_sta[state][specie].std() for specie in Species2] for state in States1}
ext_ALWC_dict = {state: [dic_grp_sta[state][specie].mean() for specie in water2] for state in States1}
ext_ALWC_std = {state: [dic_grp_sta[state][specie].std() for specie in water2] for state in States1}

ext_mix_dict = {state: [dic_grp_sta[state][specie].mean() for specie in Species3] for state in States1}
ext_mix_std = {state: [dic_grp_sta[state][specie].std() for specie in Species3] for state in States1}


def extinction_by_particle_gas():  # PG : sum of ext by particle and gas
    violin(data_set=dic_grp_sea, unit='PG')

    barplot_concen(data_set=dic_grp_sta,
                   items=['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas'],
                   labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
                   unit='Extinction')

    barplot_concen(data_set=dic_grp_Ext,
                   items=['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas'],
                   labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
                   unit='Extinction')


@set_figure(figsize=(10, 6))
def chemical_enhancement():
    fig, ax = plt.subplots()

    width = 0.20
    block = width / 4

    x = np.array([1, 2, 3, 4, 5, 6, 7])
    for i, state in enumerate(['Clean', 'Transition', 'Event']):
        val = np.array(mass_comp4_dict[state][:-1])
        std = (0,) * 7, np.array(mass_comp4_dict_std[state][:-1])

        plt.bar(x + (i + 1) * (width + block), val, yerr=std, width=width, color=Color.colors3[:-1],
                alpha=0.6 + (0.2 * i),
                edgecolor=None, capsize=None, label=state)

    ax.set_xlabel(r'$\bf Chemical\ species$')
    ax.set_xticks(x + 2 * (width + block))
    ax.set_xticklabels(['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], fontweight='normal')
    ax.set_ylabel(r'$\bf Mass\ concentration\ ({\mu}g/m^3)$')
    ax.set_ylim(0, 25)
    ax.set_title(r'$\bf Chemical\ enhancement$')

    ax.vlines(8, 0, 25, linestyles='--', colors='k')

    ax2 = ax.twinx()
    for i, state in enumerate(['Clean', 'Transition', 'Event']):
        val = np.array(mass_comp4_dict[state][-1])
        std = np.array([[0], [mass_comp4_dict_std[state][-1]]])
        plt.bar(8 + (i + 1) * (width + block), val, yerr=std, width=width, color='#96c8e6',
                alpha=0.6 + (0.2 * i),
                edgecolor=None, capsize=None, label=state)

    ax2.set_xticks(np.array([1, 2, 3, 4, 5, 6, 7, 8]) + 2 * (width + block))
    ax2.set_xticklabels(['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'ALWC'], fontweight='normal')
    ax2.set_ylabel(r'$\bf Mass\ concentration\ ({\mu}g/m^3)$')
    ax2.set_ylim(0, 100)

    a = (np.array(mass_comp4_dict['Event']) + np.array(mass_comp4_dict['Transition'])) / 2
    b = (np.array(mass_comp4_dict['Transition']) + np.array(mass_comp4_dict['Clean'])) / 2
    c = np.array(mass_comp4_dict['Event']) / np.array(mass_comp4_dict['Transition'])
    d = np.array(mass_comp4_dict['Transition']) / np.array(mass_comp4_dict['Clean'])

    for i, (posa, posb, vala, valb) in enumerate(zip(a, b, c, d)):
        if i < 7:
            ax.text(i + 1.5, posa, '{:.2f}'.format(vala), fontsize=6, weight='bold', zorder=1)
            ax.text(i + 1.25, posb, '{:.2f}'.format(valb), fontsize=6, weight='bold', zorder=1)
        else:
            ax2.text(i + 1.5, posa, '{:.2f}'.format(vala), fontsize=6, weight='bold', zorder=1)
            ax2.text(i + 1.25, posb, '{:.2f}'.format(valb), fontsize=6, weight='bold', zorder=1)

    plt.show()


def ext_mass_barplot():
    barplot_combine(data_set=ext_dry_dict,
                    data_std=ext_dry_std,
                    data_ALWC=ext_ALWC_dict,
                    data_ALWC_std=ext_ALWC_std,
                    labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'ALWC'],
                    unit='Extinction',
                    colors=Color.colors2,
                    orientation='va',
                    figsize=(12, 5))

    barplot_extend(data_set=mass_comp1_dict,
                   labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'EC'],
                   unit='PM25',
                   colors=Color.colors1,
                   orientation='va',
                   figsize=(12, 5))


def pie_plot():
    Pie.pie(data_set=mass_comp1_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='PM25', style='pie')
    Pie.pie(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction', style='donut')
    Pie.pie(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction', style='donut')
    Pie.pie(data_set=ext_amb_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'ALWC'], unit='Extinction', style='donut', colors=Color.colors2)
    Pie.donuts(data_set=mass_comp1_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='PM25')
    Pie.donuts(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction')
    Pie.donuts(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction')
    Pie.donuts(data_set=ext_amb_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'ALWC'], unit='Extinction', colors=Color.colors2)


if __name__ == '__main__':
    # pie_plot()
    ext_mass_barplot()
    # chemical_enhancement()
    extinction_by_particle_gas()
