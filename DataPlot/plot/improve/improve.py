import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from scipy.optimize import curve_fit
from DataPlot.process import *
from DataPlot.plot.core import *
from DataPlot.plot.templates import *

# TODO: this file has to be reorganized

__all__ = ['chemical_enhancement',
           'ammonium_rich',
           'pie_IMPROVE',
           'MLR_IMPROVE',
           'fRH_plot',
           'extinction_by_particle_gas'
           ]


@set_figure(figsize=(10, 6))
def chemical_enhancement(data_set: pd.DataFrame = None,
                         data_std: pd.DataFrame = None,
                         ax: Axes | None = None,
                         **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    ser_grp_sta, ser_grp_sta_std = DataClassifier(DataBase, by='State', statistic='Table')
    species = ['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'ALWC']
    data_set, data_std = ser_grp_sta.loc[:, species], ser_grp_sta_std.loc[:, species]

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


@set_figure
def ammonium_rich(df: pd.DataFrame = DataBase,
                  **kwargs) -> Axes:
    df = df[['NH4+', 'SO42-', 'NO3-', 'PM25']].dropna().copy().div([18, 96, 62, 1])
    df['required_ammonium'] = df['NO3-'] + 2 * df['SO42-']

    fig, ax = plt.subplots()

    scatter = ax.scatter(df['required_ammonium'].to_numpy(), df['NH4+'].to_numpy(), c=df['PM25'].to_numpy(),
                         vmin=0, vmax=70, cmap='jet', marker='o', s=10, alpha=1)

    ax.axline((0, 0), slope=1., color='k', lw=2, ls='--', alpha=0.5, label='1:1')
    plt.text(0.97, 0.97, r'$\bf 1:1\ Line$', color='k', ha='right', va='top', transform=ax.transAxes)

    ax.set_xlabel(r'$\bf NO_{3}^{-}\ +\ 2\ \times\ SO_{4}^{2-}\ (mole\ m^{-3})$')
    ax.set_ylabel(r'$\bf NH_{4}^{+}\ (mole\ m^{-3})$')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(ax.get_yticks())
    ax.set_title(kwargs.get('title', ''))

    color_bar = plt.colorbar(scatter, extend='both')
    color_bar.set_label(label=Unit('PM25'), size=14)

    # fig.savefig(f'Ammonium_rich_{title}')
    return ax


def pie_IMPROVE():
    Species1 = ['AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry']
    Species2 = ['AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry', 'ALWC_ext']
    Species3 = ['AS_ext', 'AN_ext', 'OM_ext', 'Soil_ext', 'SS_ext', 'EC_ext']

    ser_grp_sta, _ = DataClassifier(DataBase, by='State', statistic='Table')

    ext_dry_dict = ser_grp_sta.loc[:, Species1]
    ext_amb_dict = ser_grp_sta.loc[:, Species2]
    ext_mix_dict = ser_grp_sta.loc[:, Species3]

    Pie.donuts(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction')
    Pie.donuts(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction')
    Pie.donuts(data_set=ext_amb_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'ALWC'],
               unit='Extinction', colors=Color.colors2)


def MLR_IMPROVE(**kwargs):
    """
    Perform multiple linear regression analysis and generate plots based on IMPROVE dataset.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments for customization.

    Returns
    -------
    None

    Examples
    --------
    Example usage of MLR_IMPROVE function:

    >>> MLR_IMPROVE()

    Notes
    -----
    This function performs multiple linear regression analysis on the IMPROVE dataset and generates plots for analysis.

    - The function first selects specific species from the dataset and drops NaN values.
    - It calculates a 'Localized' value based on a multiplier and the sum of selected species.
    - Data from 'modified_IMPROVE.csv' and 'revised_IMPROVE.csv' are read and concatenated with the dataset.
    - Statistical analysis is performed using DataClassifier to calculate mean and standard deviation.
    - Plots are generated using linear_regression for Extinction vs. Revised/Modified/Localized and Pie.donuts for a
      pie chart showing the distribution of species based on Extinction.

    """
    species = ['Extinction', 'Scattering', 'Absorption',
               'total_ext_dry', 'AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry',
               'AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'OM']

    df = DataBase[species].dropna().copy()

    # multiple_linear_regression(df, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS'], y='Scattering', add_constant=True)
    # multiple_linear_regression(df, x=['POC', 'SOC', 'EC'], y='Absorption', add_constant=True)
    # multiple_linear_regression(df, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], y='Extinction', add_constant=False)

    multiplier = [2.675, 4.707, 11.6, 7.272, 0, 0.131, 10.638]
    df['Localized'] = df[['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC']].mul(multiplier).sum(axis=1)
    modify_IMPROVE = DataReader('modified_IMPROVE.csv')['total_ext_dry'].rename('Modified')
    revised_IMPROVE = DataReader('revised_IMPROVE.csv')['total_ext_dry'].rename('Revised')

    df = pd.concat([df, revised_IMPROVE, modify_IMPROVE], axis=1)

    n_df = df[['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC']].mul(multiplier)
    mean, std = DataClassifier(n_df, 'State', statistic='Table')

    ser_grp_sta, _ = DataClassifier(DataBase, by='State', statistic='Table')
    mass_comp = ser_grp_sta.loc[:, ['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC']]

    # plot
    linear_regression(df, x='Extinction', y=['Revised', 'Modified', 'Localized'], xlim=[0, 400], ylim=[0, 400],
                      regression=True, diagonal=True)
    Pie.donuts(data_set=mass_comp, labels=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'],
               unit='PM25', colors=Color.colors3)
    Pie.donuts(mean, labels=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], unit='Extinction', colors=Color.colors3)


@set_figure
def fRH_plot(**kwargs) -> Axes:
    frh = DataReader('fRH.json')

    def fitting_func(RH, a, b, c):
        f = a + b * (RH / 100) ** c
        return f

    x = frh.index.to_numpy()
    y = frh['fRHs'].to_numpy()

    popt, pcov = curve_fit(fitting_func, x, y)
    params = popt.tolist()
    val_fit = fitting_func(x, *params)

    fig, ax = plt.subplots()
    plt.plot(frh.index, frh['fRH'], 'k-o')
    plt.plot(frh.index, frh['fRHs'], 'g-o')
    plt.plot(frh.index, frh['fRHl'], 'r-o')
    plt.plot(frh.index, frh['fRHSS'], 'b-o')
    plt.xlim(0, 100)
    plt.ylim(1, )
    plt.title(r'$\bf Hygroscopic\ growth\ factor$')
    plt.grid(axis='y', color='gray', linestyle='dashed', linewidth=1, alpha=0.6)
    plt.xlabel('$RH (\\%)$')
    plt.ylabel('$f(RH)$')
    plt.legend([fr'$\bf f(RH)_{{original}}$',
                fr'$\bf f(RH)_{{small\ mode}}$',
                fr'$\bf f(RH)_{{large\ mode}}$',
                fr'$\bf f(RH)_{{sea\ salt}}$'],
               loc='upper left', prop=dict(size=16))
    # fig.savefig('fRH_plot')
    return ax


def extinction_by_particle_gas():  # PG : sum of ext by particle and gas
    ser_grp_sta, ser_grp_sta_std = DataClassifier(DataBase, by='State', statistic='Table')
    ext_particle_gas = ser_grp_sta.loc[:, ['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas']]

    Bar.barplot(data_set=ext_particle_gas, data_std=None,
                labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
                style="stacked",
                unit='Extinction',
                colors=Color.paired)

    Pie.pieplot(data_set=ext_particle_gas,
                labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
                unit='Extinction',
                style='donut',
                colors=Color.paired)


if __name__ == '__main__':
    chemical_enhancement()
    MLR_IMPROVE()
    ammonium_rich()
