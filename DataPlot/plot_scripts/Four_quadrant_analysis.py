from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot_templates import set_figure, unit, getColor
from DataPlot.data_processing import main

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = main()
subdf = df[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna()
resampled_df = subdf.resample('3H').mean()


@set_figure(figsize=(8, 6))
def four_quar(subdf):
    item = 'RH'
    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(subdf['PM25'], subdf['Vis_LPV'], s=200 * (subdf[item]/ subdf[item].max())**4, c=subdf['VC'], norm=plt.Normalize(vmin=0, vmax=2000), cmap='YlGnBu')
    axins = inset_axes(ax, width="48%", height="5%", loc=9)
    color_bar = plt.colorbar(sc, cax=axins, orientation='horizontal')
    color_bar.set_label(label=unit('VC'))

    ax.tick_params(axis='x', which='major', direction="out", length=6)
    ax.tick_params(axis='y', which='major', direction="out", length=6)
    ax.set_xlim(0., 80)
    ax.set_ylim(0., 50)
    ax.set_ylabel(r'$\bf Visibility\ (km)$')
    ax.set_xlabel(r'$\bf PM_{2.5}\ (\mu g/m^3)$')

    dot = np.linspace(subdf[item].min(), subdf[item].max(), 6).round(-1)

    for dott in dot[1:-1]:
        ax.scatter([], [], c='k', alpha=0.8, s=200 * (dott / subdf[item].max()) ** 4, label='{:.0f}'.format(dott))

    ax.legend(loc='center right', bbox_to_anchor=(0.8, 0.3, 0.2, 0.2), scatterpoints=1, frameon=False, labelspacing=0.5, title=unit('RH'))

    # fig2, ax2 = plt.subplots(1, 1)
    # sc2 = plt.scatter(PM, Est_Vis, s=30, c=VC, norm=plt.Normalize(vmin=0,vmax=1800), cmap='YlGnBu')
    # axins2 = inset_axes(ax2, width="50%", height="5%", loc=1)
    # color_bar2 = plt.colorbar(sc2, cax=axins2, orientation='horizontal')
    # color_bar2.set_label(label=r'$\bf VC\ (m^{2}/s)$')
    #
    # ax2.tick_params(axis='x', which='major',direction="out", length=6)
    # ax2.tick_params(axis='y', which='major',direction="out", length=6)
    # ax2.set_xlim(0., 80)
    # ax2.set_ylim(0., 50)
    # ax2.set_ylabel(r'$\bf Visibility\ (km)$')
    # ax2.set_xlabel(r'$\bf PM_{2.5}\ (\mu g/m^3)$')
    # plt.show()

#四象限圓餅圖-------------------------------------------------------------------------------------------------------------
for quadrant in ['1', '2', '3', '4']:
    print('Quadrant = ' + quadrant)
    for x in ['Extinction', 'Scatter', 'Absorption', 'MEE', 'MSE', 'MAE', 'VC',
              'AS_ext', 'AN_ext', 'OM_ext', 'Soil_ext', 'SS_ext', 'EC_ext',
              'AS_ext_d', 'AN_ext_d', 'OM_ext_d', 'Soil_ext_d', 'SS_ext_d', 'EC_ext_d']:
        print(x + ' = ' + '{:.2f} \u00B1 {:.2f}'.format(dic_four[quadrant][x].mean(), dic_four[quadrant][x].std()))


    sizes = [dic_four[quadrant]['AS_ext_d'].mean(), (dic_four[quadrant]['AS_ext']-dic_four[quadrant]['AS_ext_d']).mean(),
             dic_four[quadrant]['AN_ext_d'].mean(), dic_four[quadrant]['AN_ext'].mean()-dic_four[quadrant]['AN_ext_d'].mean(),
             dic_four[quadrant]['OM_ext_d'].mean(), dic_four[quadrant]['Soil_ext_d'].mean(),
             dic_four[quadrant]['SS_ext_d'].mean(), dic_four[quadrant]['SS_ext'].mean()-dic_four[quadrant]['SS_ext_d'].mean(),
             dic_four[quadrant]['EC_ext_d'].mean()]

    sizes2 = [dic_four[quadrant]['AS_ext'].mean(),
              dic_four[quadrant]['AN_ext'].mean(),
              dic_four[quadrant]['OM_ext'].mean(), dic_four[quadrant]['Soil_ext'].mean(),
              dic_four[quadrant]['SS_ext'].mean(), dic_four[quadrant]['EC_ext'].mean()]
    labels = 'Ammonium Sulfate, AS', 'Hygroscopic growth by AS', 'Ammonium Nitrate, AN', 'Hygroscopic growth by AN',\
             'Organic Matter, OM', 'Soil', 'Sea Salt, SS', 'Hygroscopic growth by SS', 'Elemental Carbon, EC'

    labels_sizes = ['{:^20}'.format(labels[0]) + '\n' + str(sizes[0]) + ' (1/Mm)',
                    '{:^20}'.format(labels[1]) + '\n' + str(sizes[1]) + ' (1/Mm)',
                    '{:^17}'.format(labels[2]) + '\n' + str(sizes[2]) + ' (1/Mm)',
                    '{:^18}'.format(labels[3]) + '\n' + str(sizes[3]) + ' (1/Mm)',
                    '{:^18}'.format(labels[4]) + '\n' + str(sizes[4]) + ' (1/Mm)',
                    '{:^20}'.format(labels[5]) + '\n' + str(sizes[5]) + ' (1/Mm)']

    colors = ['#FF3333', '#FFB5B5', '#33FF33', '#BBFFBB', '#FFFF33', '#5555FF', '#B94FFF', '#FFBFFF', '#AAAAAA']
    colors2 = ['#FF3333', '#33FF33', '#FFFF33', '#5555FF', '#B94FFF', '#AAAAAA']
    explode = (0, 0, 0, 0, 0, 0, 0, 0, 0)  # explode a slice if required 圓餅外凸
    explode2 = (0, 0, 0, 0, 0, 0)
    # Intended to serve something like a global variable
    textprops = {'fontsize': 16, 'fontname': 'Times New Roman', 'weight': 'bold'}
    prop_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    fig1, ax1 = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax1.pie(sizes, explode=explode, labels=None, colors=colors,
            autopct='%1.1f%%', shadow=False, textprops=textprops, radius=10, labeldistance=None, pctdistance=0.85,
            startangle=0, wedgeprops=dict(width=3, edgecolor='w'))

    ax1.pie(sizes2, explode=explode2, labels=None, colors=colors2,
            autopct='%1.1f%%', shadow=False, textprops=textprops, radius=13, labeldistance=None, pctdistance=0.90,
            startangle=0, wedgeprops=dict(width=3, edgecolor='w'))

    sumstr = 'Extinction' + '\n\n' + "{:.2f}".format(np.sum(sizes)) + '(1/Mm)'
    plt.text(0, 0, sumstr, horizontalalignment='center', verticalalignment='center', size=18, family='Times New Roman',
             weight='bold')

    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title(quadrant, family='Times New Roman', size=25, weight='bold')
    plt.show()


#四象限 violine----------------------------------------------------------------------------------------------------------
for col, x in zip(['MEE','MSE','MAE'], label_[-4:-2]):  #自己打要的欄位跟label
    fig, axes = plt.subplots(1, 1, figsize=(8, 5), dpi=150, constrained_layout=True)
    plt.title(col+' violin config')
    plt.xlabel('')
    plt.ylabel(x)
    violin = sns.violinplot(data=[dic_four['1'][col], dic_four['2'][col], dic_four['3'][col], dic_four['4'][col]],
                            scale='area', palette='husl', inner='quartile')
    for violin, alpha in zip(axes.collections[:], [0.8, 0.8, 0.8, 0.8]):
        violin.set_alpha(alpha)
        violin.set_edgecolor('k')

    plt.xticks([0, 1, 2, 3],
               ['High Visibility\nHigh $\mathregular{PM_{2.5}}$', 'High Visibility\nLow $\mathregular{PM_{2.5}}$',
                'Low Visibility\nLow $\mathregular{PM_{2.5}}$', 'Low Visibility\nHigh $\mathregular{PM_{2.5}}$'],
               fontsize=18, fontname="Times New Roman", weight='bold')
    plt.ylim(0, )
    Mean  = plt.scatter([0, 1, 2, 3], [dic_four['1'][col].mean(), dic_four['2'][col].mean(),
                                      dic_four['3'][col].mean(), dic_four['4'][col].mean()],
                        marker='o', s=50, facecolor="white", edgecolor="black")
    Event = plt.scatter([0, 1, 2, 3], [dic_four['1'][col].quantile(0.9), dic_four['2'][col].quantile(0.9),
                                       dic_four['3'][col].quantile(0.9), dic_four['4'][col].quantile(0.9)],
                        marker='o', s=50, facecolor="red", edgecolor="black")
    Clean = plt.scatter([0, 1, 2, 3], [dic_four['1'][col].quantile(0.1), dic_four['2'][col].quantile(0.1),
                                       dic_four['3'][col].quantile(0.1), dic_four['4'][col].quantile(0.1)],
                        marker='o', s=50, facecolor="green", edgecolor="black")
    plt.legend(handles=[Event, Mean, Clean], labels=['Event', 'Mean', 'Clean'], loc='best', prop=prop_legend)


if __name__ == '__main__':
    four_quar(resampled_df)
