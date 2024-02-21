import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import concat

from DataPlot.process import DataBase, DataReader, Classifier
from DataPlot.plot.templates import scatter, linear_regression, multiple_linear_regression
from DataPlot.plot.core import set_figure, getColor, unit


def residual_PM(_df):
    _df['residual_PM'] = _df['PM25'] - _df['AS'] - _df['AN'] - _df['OM'] - _df['SS'] - _df['EC']

    return _df[['residual_PM', 'Ti', 'Fe', 'Si']]


def residual_ext(_df):
    _df['residual_ext'] = _df['total_ext_dry'] - _df['AS_ext_dry'] - _df['AN_ext_dry'] - _df['Soil_ext_dry'] - _df[
        'SS_ext_dry']

    return _df[['residual_ext', 'POC', 'SOC']]


@set_figure(figsize=(10, 6))
def donuts_ext(data_set, labels, style='donut', title='', symbol=True):
    prop_legend = {'size': 12, 'family': 'Times New Roman', 'weight': 'bold'}
    textprops = {'fontsize': 14, 'fontfamily': 'Times New Roman', 'fontweight': 'bold'}

    labels = 'AS', 'AN', 'POA', 'SOA', 'EC'

    values1 = np.array(list(data_set.values()))[3]
    values2 = np.array(list(data_set.values()))[2]
    values3 = np.array(list(data_set.values()))[1]

    colors1 = getColor(kinds='3-4')

    def adjust_opacity(color, alpha):
        # 將顏色轉換為RGB表示
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        # 調整透明度
        r_new = int(alpha * r + (1 - alpha) * 255)
        g_new = int(alpha * g + (1 - alpha) * 255)
        b_new = int(alpha * b + (1 - alpha) * 255)
        # 轉換為新的色碼
        new_color = '#{:02X}{:02X}{:02X}'.format(r_new, g_new, b_new)
        return new_color

    colors2 = [adjust_opacity(color, 0.8) for color in colors1]
    colors3 = [adjust_opacity(color, 0.6) for color in colors1]

    fig, ax = plt.subplots(1, 1)
    ax.pie(values1, labels=None, colors=colors1, textprops=textprops,
           autopct='%1.1f%%',
           pctdistance=0.9, radius=14, wedgeprops=dict(width=3, edgecolor='w'))

    ax.pie(values2, labels=None, colors=colors2, textprops=textprops,
           autopct='%1.1f%%',
           pctdistance=0.85, radius=11, wedgeprops=dict(width=3, edgecolor='w'))

    ax.pie(values3, labels=None, colors=colors3, textprops=textprops,
           autopct='%1.1f%%',
           pctdistance=0.80, radius=8, wedgeprops=dict(width=3, edgecolor='w'))

    text = 'Average (1/Mm)' + '\n\n' + 'Event : ' + "{:.2f}".format(np.sum(values1)) + '\n' + \
           'Transition : ' + "{:.2f}".format(np.sum(values2)) + '\n' + \
           'Clean : ' + "{:.2f}".format(np.sum(values3))

    ax.text(0, 0, text, fontdict=textprops, ha='center', va='center')
    ax.axis('equal')
    ax.set_title(f'{title}', size=20, weight='bold')

    ax.legend(labels, loc='center', prop=prop_legend, title_fontproperties=dict(weight='bold'),
              title='Outer : Event' + '\n' + 'Middle : Transition' + '\n' + 'Inner : Clean',
              bbox_to_anchor=(0.8, 0, 0.5, 1), frameon=False)
    plt.show()
    # fig.savefig(f"IMPROVE_ext_donuts_{title}", transparent=True)


def MLR_IMPROVE():
    df = DataBase
    dic_grp_sta = Classifier(df, 'state')

    species = ['Extinction', 'Scattering', 'Absorption', 'total_ext_dry', 'AS_ext_dry', 'AN_ext_dry',
               'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry',
               'AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'OM']

    df_cal = df[species].dropna().copy()

    multiple_linear_regression(df_cal, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS'], y='Scattering', add_constant=True)
    multiple_linear_regression(df_cal, x=['POC', 'SOC', 'EC'], y='Absorption', add_constant=True)
    multiple_linear_regression(df_cal, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], y='Extinction', add_constant=True)

    df_cal['Localized'] = df_cal[['AS', 'AN', 'POC', 'SOC', 'EC']].mul([2.74, 4.41, 11.5, 7.34, 12.27]).sum(axis=1)
    modify_IMPROVE = DataReader('modify_IMPROVE.csv')['total_ext_dry'].rename('Modified')
    revised_IMPROVE = DataReader('revised_IMPROVE.csv')['total_ext_dry'].rename('Revised')

    df = concat([df_cal, revised_IMPROVE, modify_IMPROVE], axis=1)

    n_df = df_cal[['AS', 'AN', 'POC', 'SOC', 'EC']].mul([2.74, 4.41, 11.5, 7.34, 12.27])
    new_df = concat([df_cal['Extinction'], n_df], axis=1)
    new_dic = Classifier(new_df, 'state')

    ext_dry_dict = {state: [new_dic[state][specie].mean() for specie in ['AS', 'AN', 'POC', 'SOC', 'EC']]
                    for state in ['Total', 'Clean', 'Transition', 'Event']}

    # plot
    linear_regression(df, x='Extinction', y=['Revised', 'Modified', 'Localized'], xlim=[0, 400], ylim=[0, 400],
                      regression=True, diagonal=True)
    donuts_ext(ext_dry_dict, labels=['AS', 'AN', 'POC', 'SOC', 'EC'])


if __name__ == '__main__':
    MLR_IMPROVE()
