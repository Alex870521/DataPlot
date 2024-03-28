import numpy as np
import pandas as pd
from DataPlot import *


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
    plot.templates.Violin.violin(data_set=dic_grp_sea,
                                 unit='PG')

    plot.templates.Bar.barplot(data_set=ext_particle_gas, data_std=None,
                               labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
                               display="stacked",
                               unit='Extinction')

    plot.templates.Bar.barplot(data_set=ext_dry_dict, data_std=ext_dry_std,
                               labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'EC'],
                               display="dispersed",
                               unit='Extinction')


def pie_plot():
    plot.templates.Pie.pieplot(data_set=mass_comp1_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='PM25',
                               style='pie')
    plot.templates.Pie.pieplot(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction',
                               style='donut')
    plot.templates.Pie.pieplot(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction',
                               style='donut')
    plot.templates.Pie.pieplot(data_set=ext_amb_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'ALWC'],
                               unit='Extinction',
                               style='donut', colors=Color.colors2)

    plot.templates.Pie.donuts(data_set=mass_comp3_dict, labels=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'BC', 'Water'],
                              unit='PM25',
                              colors=Color.colors3)
    plot.templates.Pie.donuts(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction')
    plot.templates.Pie.donuts(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], unit='Extinction')
    plot.templates.Pie.donuts(data_set=ext_amb_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'ALWC'],
                              unit='Extinction',
                              colors=Color.colors2)


def four_quar():
    subdf = DataBase[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna().resample('3h').mean()
    plot.templates.scatter(subdf, x='PM25', y='Vis_LPV', c='VC', s='RH', cmap='YlGnBu', fig_kws={}, plot_kws={})
