import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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


def MLR_IMPROVE():
    species = ['Extinction', 'Scattering', 'Absorption', 'total_ext_dry', 'AS_ext_dry', 'AN_ext_dry',
               'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry',
               'AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'OM']

    df = DataBase[species].dropna().copy()

    # multiple_linear_regression(df, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS'], y='Scattering', add_constant=True)
    # multiple_linear_regression(df, x=['POC', 'SOC', 'EC'], y='Absorption', add_constant=True)
    # multiple_linear_regression(df, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], y='Extinction', add_constant=True)

    multiplier = [2.675, 4.707, 11.6, 7.272, 0, 0.131, 10.638]
    df['Localized'] = df[['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC']].mul(multiplier).sum(axis=1)
    modify_IMPROVE = DataReader('modified_IMPROVE.csv')['total_ext_dry'].rename('Modified')
    revised_IMPROVE = DataReader('revised_IMPROVE.csv')['total_ext_dry'].rename('Revised')

    df = concat([df, revised_IMPROVE, modify_IMPROVE], axis=1)

    n_df = df[['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC']].mul(multiplier)
    mean, std = DataClassifier(n_df, 'State', statistic='Table')

    # plot
    linear_regression(df, x='Extinction', y=['Revised', 'Modified', 'Localized'], xlim=[0, 400], ylim=[0, 400],
                      regression=True, diagonal=True)
    Pie.donuts(mean, labels=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], unit='Extinction', colors=Color.colors3)


frh = DataReader('fRH.json')

@set_figure
def fRH_plot() -> plt.Axes:
    print('Plot: fRH_plot')

    fig, ax = plt.subplots( figsize=(6, 6))
    plt.plot(frh.index, frh['fRH'], 'k-o', lw=2)
    plt.plot(frh.index, frh['fRHs'], 'g-o', lw=2)
    plt.plot(frh.index, frh['fRHl'], 'r-o', lw=2)
    plt.plot(frh.index, frh['fRHSS'], 'b-o', lw=2)
    plt.xlim(0, 100)
    plt.ylim(1, )
    plt.title(r'$\bf Hygroscopic\ growth\ factor$')
    plt.grid(axis='y', color='gray', linestyle='dashed', linewidth=1, alpha=0.6)
    plt.xlabel(r'$\bf RH\ (\%)$')
    plt.ylabel(r'$\bf f(RH)$')
    plt.legend([fr'$\bf f(RH)_{{original}}$',
                fr'$\bf f(RH)_{{small\ mode}}$',
                fr'$\bf f(RH)_{{large\ mode}}$',
                fr'$\bf f(RH)_{{sea\ salt}}$'],
               loc='upper left', prop=dict(size=16))
    # fig.savefig('fRH_plot')
    return ax


@set_figure
def fRH_fit():
    split = [36, 75]
    x1 = frh.index.to_numpy()[split[0]:split[1]]
    y1 = frh['fRHs'].values[split[0]:split[1]]
    x2 = frh.index.to_numpy()[split[1]:]
    y2 = frh['fRHs'].values[split[1]:]

    def f__RH(RH, a, b, c):
        f = a + b * (RH/100)**c
        return f

    popt, pcov = curve_fit(f__RH, x1, y1)
    a, b, c = popt
    yvals = f__RH(x1, a, b, c) #擬合y值
    print(u'係數a:', a)
    print(u'係數b:', b)
    print(u'係數c:', c)

    popt2, pcov2 = curve_fit(f__RH, x2, y2)
    a2, b2, c2 = popt2
    yvals2 = f__RH(x2, a2, b2, c2) #擬合y值
    print(u'係數a2:', a2)
    print(u'係數b2:', b2)
    print(u'係數c2:', c2)

    fig, axes = plt.subplots(figsize=(5, 5))
    plt.scatter(x1, y1, label='original values')
    plt.plot(x1, yvals, 'r', label='polyfit values')
    plt.scatter(x2, y2, label='original values')
    plt.plot(x2, yvals2, 'r', label='polyfit values')
    plt.xlabel(r'$\bf RH$')
    plt.ylabel(r'$\bf f(RH)$')
    plt.legend(loc='best')
    plt.title(r'$\bf Curve fit$')
    plt.show()


@set_figure
def ammonium_rich(_df: pd.DataFrame, **kwargs) -> plt.Axes:
    print('Plot: ammonium_rich')
    df = _df[['NH4+', 'SO42-', 'NO3-', 'PM25']].dropna().copy().div([18, 96, 62, 1])
    df['required_ammonium'] = df['NO3-'] + 2 * df['SO42-']

    fig, ax = plt.subplots()

    scatter = ax.scatter(df['required_ammonium'], df[['NH4+']], c=df[['PM25']].values,
                         vmin=0, vmax=70, cmap='jet', marker='o', s=10, alpha=1)
    ax.axline((0, 0), slope=1., color='r', lw=3, ls='--', label='1:1')
    ax.set_xlabel(r'$\bf NO_{3}^{-}\ +\ 2\ \times\ SO_{4}^{2-}\ (mole\ m^{-3})$')
    ax.set_ylabel(r'$\bf NH_{4}^{+}\ (mole\ m^{-3})$')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(ax.get_yticks())
    ax.set_title(kwargs.get('title', ''))

    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')

    color_bar = plt.colorbar(scatter, extend='both')
    color_bar.set_label(label=Unit('PM25'), size=14)

    # fig.savefig(f'Ammonium_rich_{title}')
    return ax


if __name__ == '__main__':
    # pie_plot()
    # chemical_enhancement()
    extinction_by_particle_gas()
