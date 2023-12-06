import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataPlot.Data_processing import main
from DataPlot.plot_templates import set_figure, unit, getColor
from DataPlot.plot_templates import scatter, violin, pie_ext

from DataPlot.Data_processing.Data_classify import state_classify, season_classify

prop_legend = {'size': 12, 'family': 'Times New Roman', 'weight': 'bold'}
textprops = {'fontsize': 14, 'fontfamily': 'Times New Roman', 'fontweight': 'bold'}

colors1 = getColor(kinds='3-3')


def inner_pct(pct, symbol=True):
    if symbol:
        if pct < 8:
            return ''
        else:
            return '{:.1f}%'.format(pct)
    else:
        return ''


def outer_pct(pct, symbol=True):
    if symbol:
        if pct > 8:
            return ''
        else:
            return '{:.1f}%'.format(pct)
    else:
        return ''


if __name__ == '__main__':
    print('--- building data ---')
    df = main()['2020-10-01':'2021-05-06']

    dic_grp_sta = state_classify(df)
    dic_grp_sta['Clean']['fRH_PNSD'].mean()
    dic_grp_sta['Transition']['fRH_PNSD'].mean()
    dic_grp_sta['Event']['fRH_PNSD'].mean()
    # scatter(df, x='RH', y='ALWC', c='PM25', y_range=[0, 200], c_range=[0, 60])
    # scatter(df, x='RH', y='ALWC', c='Extinction', y_range=[0, 200], c_range=[0, 200])
    # scatter(df, x='RH', y='gRH', c='ALWC_mass_ratio', y_range=[1, 3], c_range=[0, 4])
    tar = 'kappa'
    a = dic_grp_sta['Event'][tar].dropna().values
    b = dic_grp_sta['Transition'][tar].dropna().values
    c = dic_grp_sta['Clean'][tar].dropna().values
    # violin([1.38, 1.31, 1.36], [a, b, c], title='')

    Species3 = ['AS_ext', 'AN_ext', 'OM_ext', 'Soil_ext', 'SS_ext', 'EC_ext']
    items = ['AS_mass', 'AN_mass', 'OM_mass', 'Soil_mass', 'SS_mass', 'EC_mass', 'gRH']
    States1 = ['Total', 'Clean', 'Transition', 'Event']


    def RH_based():
        bins = np.array([0, 40, 60, 80, 100])
        labels = ['0-40', '40~60', '60~80', '80~100']
        df['RH_cut'] = pd.cut(df['RH'], bins, labels=labels)
        df_RH_group = df.groupby('RH_cut')
        dic = {}
        for _grp, _df in df_RH_group:
            print(_df['gRH'].mean(), _df['gRH'].std())
            # dic_grp_sta = state_classify(_df)
            # dic[_grp] = {state: [dic_grp_sta[state][specie].mean() for specie in Species3] for state in States1}
            dic[_grp] = [_df[specie].mean() for specie in items]
        return dic


    dic = RH_based()


    @set_figure
    def dyr_ALWC_gRH(data_set, labels, gRH=1, title='', symbol=True):
        label_colors = colors1

        radius = 4
        width = 4

        pct_distance = 0.6

        fig, ax = plt.subplots(1, 1, figsize=(4 * gRH, 4), dpi=150)

        ax.pie(data_set, labels=None, colors=label_colors, textprops=textprops,
               autopct=lambda pct: inner_pct(pct, symbol=symbol),
               pctdistance=pct_distance, radius=radius, wedgeprops=dict(width=width, edgecolor='w'))

        ax.pie(data_set, labels=None, colors=label_colors, textprops=textprops,
               autopct=lambda pct: outer_pct(pct, symbol=symbol),
               pctdistance=1.2, radius=radius, wedgeprops=dict(width=width, edgecolor='w'))

        ax.pie([100, ], labels=None, colors=[label_colors[-1]] * 1, textprops=textprops,
               autopct=lambda pct: outer_pct(pct, symbol=False),
               pctdistance=1.2, radius=radius * gRH, wedgeprops=dict(width=radius * (gRH - 1), edgecolor='w'))

        ax.axis('equal')
        ax.set_title(rf'$\bf {title}$')
        fig.savefig(f'gRH_{title}', transparent=True)
        plt.show()


    for group, values in dic.items():

        dyr_ALWC_gRH(values[:-1], gRH=values[-1], labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC', 'ALWC'], title=group)

    # for label in labels:
    #     piePlot.pie_ext(data_set=dic[label],
    #                     labels=['AS', 'AN', 'OM', 'Soil', 'SS', 'BC'], style='donut',
    #                     title=label)
