import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataPlot.process import *
from DataPlot.plot.core import *
from DataPlot.plot.templates import *

DataBase['0'] = 0
Species1 = ['AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry']
Species2 = ['AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry', 'ALWC_ext']
Species3 = ['AS_ext', 'AN_ext', 'OM_ext', 'Soil_ext', 'SS_ext', 'EC_ext']
water = ['ALWC_AS_ext', 'ALWC_AN_ext', '0', '0', 'ALWC_SS_ext', '0']

mass_1 = ['AS', 'AN', 'OM', 'Soil', 'SS', 'EC']
mass_2 = ['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'ALWC']
mass_3 = ['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'ALWC']

dic_grp_sea = DataClassifier(DataBase, by='Season', statistic='Dict')
dic_grp_sta = DataClassifier(DataBase, by='State', statistic='Dict')
dic_grp_Ext = DataClassifier(DataBase, by='Extinction', statistic='Dict', cut_bins=np.linspace(0, 400, 11))
dic_grp_RH = DataClassifier(DataBase, by='RH', statistic='Dict', cut_bins=np.array([20, 40, 60, 80, 100]))

ser_grp_sta, ser_grp_sta_std = DataClassifier(DataBase, by='State', statistic='Table')
ser_grp_Ext, ser_grp_Ext_sed = DataClassifier(DataBase, by='Extinction', statistic='Table',
                                              cut_bins=np.linspace(0, 400, 11))

mass_comp1_dict, _ = ser_grp_sta.loc[:, mass_1], ser_grp_sta_std.loc[:, mass_1]
mass_comp3_dict, mass_comp3_dict_std = ser_grp_sta.loc[:, mass_3], ser_grp_sta_std.loc[:, mass_3]

ext_dry_dict, ext_dry_std = ser_grp_sta.loc[:, Species1], ser_grp_sta_std.loc[:, Species1]
ext_amb_dict, ext_amb_std = ser_grp_sta.loc[:, Species2], ser_grp_sta_std.loc[:, Species2]
ext_mix_dict, ext_mix_std = ser_grp_sta.loc[:, Species3], ser_grp_sta_std.loc[:, Species3]
ext_ALWC_dict, ext_ALWC_std = ser_grp_sta.loc[:, water], ser_grp_sta_std.loc[:, water]

ext_particle_gas = ser_grp_Ext.loc[:, ['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas']]


def extinction_by_particle_gas():  # PG : sum of ext by particle and gas
    Violin.violin(data_set=dic_grp_sea,
                  unit='PG')

    Bar.barplot(data_set=ext_particle_gas, data_std=None,
                labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
                display="stacked",
                unit='Extinction')

    Bar.barplot(data_set=ext_dry_dict, data_std=ext_dry_std,
                labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'EC'],
                display="dispersed",
                unit='Extinction')


@set_figure(figsize=(10, 6))
def chemical_enhancement(data_set: pd.DataFrame = mass_comp3_dict,
                         data_std: pd.DataFrame = mass_comp3_dict_std,
                         ax: plt.Axes | None = None,
                         **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    width = 0.20
    block = width / 4

    x = np.array([1, 2, 3, 4, 5, 6, 7])
    for i, state in enumerate(['Clean', 'Transition', 'Event']):
        val = np.array(data_set.iloc[i, :-1])
        std = (0,) * 7, np.array(data_std.iloc[i, :-1])

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
        val = np.array(data_set.iloc[i, -1])
        std = np.array([[0], [data_std.iloc[i, -1]]])
        plt.bar(8 + (i + 1) * (width + block), val, yerr=std, width=width, color='#96c8e6',
                alpha=0.6 + (0.2 * i),
                edgecolor=None, capsize=None, label=state)

    ax2.set_xticks(np.array([1, 2, 3, 4, 5, 6, 7, 8]) + 2 * (width + block))
    ax2.set_xticklabels(['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'ALWC'], fontweight='normal')
    ax2.set_ylabel(r'$\bf Mass\ concentration\ ({\mu}g/m^3)$')
    ax2.set_ylim(0, 100)

    a = (np.array(data_set.loc['Event']) + np.array(data_set.loc['Transition'])) / 2
    b = (np.array(data_set.loc['Transition']) + np.array(data_set.loc['Clean'])) / 2
    c = np.array(data_set.loc['Event']) / np.array(data_set.loc['Transition'])
    d = np.array(data_set.loc['Transition']) / np.array(data_set.loc['Clean'])

    for i, (posa, posb, vala, valb) in enumerate(zip(a, b, c, d)):
        if i < 7:
            ax.text(i + 1.5, posa, '{:.2f}'.format(vala), fontsize=6, weight='bold', zorder=1)
            ax.text(i + 1.25, posb, '{:.2f}'.format(valb), fontsize=6, weight='bold', zorder=1)
        else:
            ax2.text(i + 1.5, posa, '{:.2f}'.format(vala), fontsize=6, weight='bold', zorder=1)
            ax2.text(i + 1.25, posb, '{:.2f}'.format(valb), fontsize=6, weight='bold', zorder=1)

    plt.show()


def pie_plot():
    Pie.pieplot(data_set=mass_comp1_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='PM25', style='pie')
    Pie.pieplot(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction', style='donut')
    Pie.pieplot(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction', style='donut')
    Pie.pieplot(data_set=ext_amb_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'ALWC'], unit='Extinction',
                style='donut', colors=Color.colors2)

    Pie.donuts(data_set=mass_comp3_dict, labels=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'BC', 'Water'], unit='PM25',
               colors=Color.colors3)
    Pie.donuts(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction')
    Pie.donuts(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction')
    Pie.donuts(data_set=ext_amb_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'ALWC'], unit='Extinction',
               colors=Color.colors2)


if __name__ == '__main__':
    # pie_plot()
    # chemical_enhancement()
    extinction_by_particle_gas()
