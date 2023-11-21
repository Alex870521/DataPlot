from pathlib import Path
from scipy.optimize import curve_fit
from Data_processing import integrate
from Data_classify import state_classify, season_classify, Seasons
from config.custom import setFigure, unit, getColor

from config.violinPlot import violin
from config.barPlot import barplot_concen, barplot_combine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from config.scatterPlot import scatter, scatter_mutiReg, box_scatter
import config.piePlot as piePlot
import config.barPlot as barPlot
import config.violinPlot as violinPlot

colors1 = getColor(kinds='1')
colors2 = getColor(kinds='2')
colors3 = getColor(kinds='3')
colors3b = getColor(kinds='3-3')
colors3c = ['#A65E58', '#A5BF6B', '#a6710d', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB', '#96c8e6']
colors4 = getColor(kinds='4')
colors5 = getColor(kinds='5')

df = integrate()['2020-09-04':'2021-05-06']
dic_grp_sea = season_classify(df)
dic_grp_sta = state_classify(df)

Species1 = ['AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry']
Species2 = ['AS_ext_dry', 'ALWC_AS_ext', 'AN_ext_dry', 'ALWC_AN_ext', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'ALWC_SS_ext', 'EC_ext_dry']
Species3 = ['AS_ext', 'AN_ext', 'OM_ext', 'Soil_ext', 'SS_ext', 'EC_ext']

dry_particle = ['AS_ext_dry', 'AN_ext_dry', 'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry']
water = ['ALWC_ext']
water2 = ['ALWC_AS_ext', 'ALWC_AN_ext', 'ALWC_SS_ext']

mass_1 = ['AS_mass', 'AN_mass', 'OM_mass', 'Soil_mass', 'SS_mass', 'EC_mass']
mass_2 = ['AS_mass', 'AN_mass', 'OM_mass', 'Soil_mass', 'SS_mass', 'EC_mass', 'others_mass']
mass_3 = ['AS_mass', 'AN_mass', 'OM_mass', 'Soil_mass', 'SS_mass', 'EC_mass', 'ALWC']
mass_4 = ['AS_mass', 'AN_mass', 'POC_mass', 'SOC_mass', 'Soil_mass', 'SS_mass', 'EC_mass', 'ALWC']

States1 = ['Total', 'Clean', 'Transition', 'Event']


if __name__ == '__main__':
    mass_comp1_dict = {state: [dic_grp_sta[state][specie].mean() for specie in mass_1] for state in States1}
    mass_comp2_dict = {state: [dic_grp_sta[state][specie].mean() for specie in mass_2] for state in States1}
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

    ALWC = {state: dic_grp_sta[state][water].mean().sum() for state in States1}
    Dry_part = {state: dic_grp_sta[state][dry_particle].mean().sum() for state in States1}


    def extinction_by_particle_gas():  # PG : sum of ext by particle and gas
        def total_light_ext_violin():
            PG_means = [df['PG'].mean()] + [dic_grp_sea[season]['Total']['PG'].mean() for season in Seasons]
            PG_vals = [df['PG'].dropna().values] + [dic_grp_sea[season]['Total']['PG'].dropna().values for season in
                                                    Seasons]
            ticks = ['Total', '2020 \n Summer  ', '2020 \n Autumn  ', '2020 \n Winter  ', '2021 \n Spring  ']
            violin(means=PG_means, data_set=PG_vals, ticks=ticks, title='Distribution pattern of light extinction')

        def total_light_ext_barplot():
            dic_grp_Ext = {}
            dic_grp_Ext['Total'] = df[['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas']].mean().values
            for key, _df in dic_grp_sea.items():
                dic_grp_Ext[key] = _df['Total'][['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas']].mean().values

            barplot_concen(data_set=dic_grp_Ext,
                           labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
                           title='Title')

        # total_light_ext_violin()
        # total_light_ext_barplot()

    # 以EXT
    def Ext_based():
        bins = np.linspace(0, 400, 11)
        labels = (bins + (bins[1] - bins[0]) / 2)[:-1]
        target_df = dic_grp_sta['Total']
        target_df['Ext_qcut'] = pd.cut(target_df['Extinction'], bins, labels=labels)
        df_group_Ext = target_df.groupby('Ext_qcut')
        dic_grp_Ext = {}
        dic_grp_mean = {}
        dic_grp_std = {}
        for _grp, _df in df_group_Ext:
            dic_grp_Ext[int(_grp)] = _df[
                ['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas']].mean().values
            dic_grp_mean[_grp] = _df[['VC', 'PM25']].mean().values
            dic_grp_std[_grp] = _df[['VC', 'PM25']].std().values

        barplot_concen(data_set=dic_grp_Ext,
                       labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
                       title='TRENDS',
                       )

        @setFigure(figsize=(8, 6))
        def aaaa():
            ext = np.array(list(dic_grp_mean.keys()))
            val = np.array(list(dic_grp_mean.values()))
            std = np.array(list(dic_grp_std.values()))

            fig, ax = plt.subplots(1, 1)
            line1, _, __ = ax.errorbar(ext, val[:, 0], std[:, 0], color='#115162', linestyle='None', marker='o',
                                       label='$ VC (m^2/s)$')
            ax.fill_between(x=ext, y1=val[:, 0] - std[:, 0], y2=val[:, 0] + std[:, 0])
            # line1, = ax.config(ext, val[:, 0], 'o-', linewidth=2, color='#115162', label=r'$\rm VC$')
            ax.set_xticks(bins)
            ax.set_xlabel(unit('Extinction'))
            ax.set_ylabel(unit('VC'))
            ax.tick_params(axis='y', colors=line1.get_color())
            ax.yaxis.label.set_color(line1.get_color())
            ax.spines['right'].set_color(line1.get_color())
            handles, labels = ax.get_legend_handles_labels()

            ax2 = ax.twinx()
            line2, _, __ = ax2.errorbar(ext, val[:, 1], std[:, 1], color='#7FAE80', linestyle='None', marker='o',
                                        label='$PM_{2.5}$')
            # line2, = ax2.config(ext, val[:, 1], 'o-', linewidth=2, color='#7FAE80', label='$PM_{2.5}$')
            ax2.set_ylabel(unit('PM25'))
            ax2.tick_params(axis='y', colors=line2.get_color())
            ax2.yaxis.label.set_color(line2.get_color())
            ax2.spines['right'].set_color(line2.get_color())
            handles2, labels2 = ax2.get_legend_handles_labels()

            bbox = (0.1, 1.1)
            ax.legend([handles[0], handles2[0]], [r'$\bf VC (m^2/s)$', r'$\bf PM_{2.5}$'], loc='upper left',
                      bbox_to_anchor=bbox, frameon=False, ncol=2)
            plt.show()

        # aaaa()

    # 以RH分類畫圖
    def RH_based():
        bins = np.array([0, 40, 60, 80, 100])
        labels = ['0~40', '40~60', '60~80', '80~100']
        df['RH_cut'] = pd.cut(df['RH'], bins, labels=labels)
        df_RH_group = df.groupby('RH_cut')
        dic = {}
        for _grp, _df in df_RH_group:
            dic_grp_sta = state_classify(_df)
            dic[_grp] = {state: [dic_grp_sta[state][specie].mean() for specie in Species3] for state in States1}

        for label in labels:
            piePlot.pie_ext(data_set=dic[label],
                            labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], style='donut',
                            title=label)


    @setFigure(figsize=(10, 6))
    def chemical_enhancement():
        fig, ax = plt.subplots(1, 1)

        width = 0.20
        block = width / 4
        leg_cont = []
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        for i, state in enumerate(['Clean', 'Transition', 'Event']):
            val = np.array(mass_comp4_dict[state])
            std = (0,) * 8, np.array(mass_comp4_dict_std[state])

            _ = plt.bar(x + (i + 1) * (width + block), val, yerr=std, width=width, color=colors3c,
                        alpha=0.6 + (0.2 * i),
                        edgecolor=None, capsize=None, label=state)
            leg_cont.append(_)

        ax.set_xlabel(r'$\bf Chemical\ species$')
        ax.set_xticks(x + (2) * (width + block))
        ax.set_xticklabels(['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'ALWC'], fontweight='normal')
        ax.set_ylabel(r'$\bf Mass\ concentration\ ({\mu}g/m^3)$')
        ax.set_ylim(0, 25)
        ax.set_title(r'$\bf Chemical\ enhancement$')
        # ax.legend(leg_cont, ['Clean', 'Transition', 'Event'])

        ax2 = ax.twinx()
        # for i, state in enumerate(['Clean', 'Transition', 'Event']):
        #
        #     val = np.array(mass_comp4_dict[state])
        #     std = (0,) * 8, np.array(mass_comp4_dict_std[state])
        #
        #     _ = ax2.bar(x + (i + 1) * (width + block), val, yerr=std, width=width, color=colors3c, alpha=0.6+(0.2*i),
        #                 edgecolor=None, capsize=None, label=state)
        #     leg_cont.append(_)

        ax2.set_xticks(x + (2) * (width + block))
        ax2.set_ylabel(r'$\bf Mass\ concentration\ ({\mu}g/m^3)$')
        ax2.set_ylim(0, 90)

        a = np.array(mass_comp4_dict['Event']) / np.array(mass_comp4_dict['Transition'])
        b = np.array(mass_comp4_dict['Transition']) / np.array(mass_comp4_dict['Clean'])

        ax3 = ax.twinx()
        ax3.spines.right.set_position(("axes", 1.12))
        point1 = ax3.scatter(x + (2.5) * (width + block), a, s=30, color='white', edgecolor='k', marker='o',
                             label='Event / Transition')
        point2 = ax3.scatter(x + (1.5) * (width + block), b, s=30, color='white', edgecolor='k', marker='s',
                             label='Transition / Clean')
        ax3.set_ylabel(r'$\bf Enhancement\ ratio$')
        ax3.set_ylim(0, 4)
        plt.legend(handles=[point1, point2], loc='upper left', prop=dict(weight='bold',))
        plt.show()


    def ext_mass_barplot():
        barPlot.barplot_combine(data_set=ext_dry_dict,
                                data_std=ext_dry_std,
                                data_ALWC=ext_ALWC_dict,
                                data_ALWC_std=ext_ALWC_std,
                                labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'ALWC'],
                                colors=colors3b,
                                title='',
                                orientation='va',
                                figsize=(12, 12))

        # barPlot.barplot_extend(data_set=mass_comp1_dict,
        #                        labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'other'],
        #                        colors=colors2,
        #                        title='Total',
        #                        orientation='va',
        #                        figsize=(12, 5))


    def pie_plot():
        # piePlot.pie_mass(data_set=mass_comp2_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'Others'], style='donut', title='PM')
        # piePlot.donuts_mass(data_set=mass_comp2_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'Others'], title='Dry')
        # piePlot.pie_ext(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], style='donut', title='Dry')
        # piePlot.pie_ext(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], style='donut', title='Ambient')
        # piePlot.pie_ext(data_set=ext_amb_dict, labels=['AS', 'AS_ALWC', 'AN', 'AN_ALWC', 'OM', 'Soil', 'SS', 'SS_ALWC', 'BC'], style='donut', title='Mix')
        # piePlot.donuts_ext(data_set=ext_dry_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], title='Dry')
        # piePlot.donuts_ext(data_set=ext_mix_dict, labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], title='Ambient')
        piePlot.donuts_ext(data_set=ext_amb_dict, labels=['AS', 'AS_ALWC', 'AN', 'AN_ALWC', 'OM', 'Soil', 'SS', 'SS_ALWC', 'EC'], title='Ambient')


    # pie_plot()
    # ext_mass_barplot()
    # chemical_enhancement()
    # extinction_by_particle_gas()
    # RH_based()
    # Ext_based()

    # for items in ['Extinction', 'Scattering', 'Absorption', 'SSA',
    #               'MEE', 'MSE', 'MAE', 'PM1', 'PM25', 'WS', 'PBLH',
    #               'VC', 'AT', 'RH', 'ALWC']:
    #     print(f'{items}: ' + str(df[items].mean().__round__(2)) + ' + ' + str(df[items].std().__round__(2)))

    # for state in States1:
    #     print(state)
    #     for items in ['Extinction', 'Scattering', 'Absorption', 'SSA',
    #                   'MEE', 'MSE', 'MAE', 'PM1', 'PM25', 'WS', 'PBLH',
    #                   'VC', 'AT', 'RH', 'ALWC']:
    #
    #         print(f'{items}: ' + str(dic_grp_sta[state][items].mean().__round__(2)) + ' + ' + str(
    #             dic_grp_sta[state][items].std().__round__(2)))


    # for season in Seasons:
    #     # unclass.ammonium_rich(dic_grp_sea[season]['Total'], title=season)
    #     for items in ['Extinction', 'Scattering', 'Absorption', 'SSA',
    #                   'MEE', 'MSE', 'MAE', 'PM1', 'PM25', 'WS', 'PBLH',
    #                   'VC', 'AT', 'RH', 'ALWC']:
    #         print(season)
    #         print(f'{items}: ' + str(dic_grp_sea[season]['Total'][items].mean().__round__(2)) + ' + ' + str(
    #             dic_grp_sea[season]['Total'][items].std().__round__(2)))

    # Extinction CDF to define the event
    # st_tm, fn_tm = pd.Timestamp('2020-09-04'), pd.Timestamp('2021-05-06')
    # data = df.loc[st_tm:fn_tm].Extinction.dropna().values
    #
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150, constrained_layout=True)
    # count, bins_count = np.histogram(data, bins=50)
    #
    # pdf = count / sum(count)
    # cdf = np.cumsum(pdf)
    #
    # plt.config(bins_count[1:], pdf, label="CDF")
    # plt.xlabel('a')
    # plt.ylabel(r'$\bf Cumulative\ Distribution$')
    # plt.xlim(0,)
    # plt.ylim(0,)

    # for state in States1:
    #     print(state)
    #     print(dic_grp_sta[state]['Bext_dry'].mean(), dic_grp_sta[state]['Bext_dry'].std())
    #     print(dic_grp_sta[state]['Bext'].mean(), dic_grp_sta[state]['Bext'].std())
    #     print(dic_grp_sta[state]['MEE_dry_PNSD'].mean(), dic_grp_sta[state]['MEE_dry_PNSD'].std())
    #     print(dic_grp_sta[state]['MEE_PNSD'].mean(), dic_grp_sta[state]['MEE_PNSD'].std())
    # for state in ['Clean', 'Event']:
        # aaa = dic_grp_sta[state]
        # scatter(aaa, x='Extinction', y='MSE', c='OM_mass_ratio', y_range=[0, 10], c_range=[0, 0.5], title=state)
        # scatter(aaa, x='GMDs', y='MSE', c='AS_mass_ratio', y_range=[0, 10], c_range=[0, 0.4], title=state)
        # scatter(aaa, x='GMDs', y='MSE', c='AN_mass_ratio', y_range=[0, 10], c_range=[0, 0.4], title=state)
        # scatter(aaa, x='Ox', y='SOC_mass', y_range=[0, 10], title=state)
    box_scatter(df, x='Extinction', y='SSA', c='EC_mass_ratio', s='PM25', y_range=[0, 1], c_range=[0, 0.07], title="")