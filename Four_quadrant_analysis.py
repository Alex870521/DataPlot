from os.path import join as pth
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from config.custom import setFigure, unit, getColor
from data_processing import integrate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = integrate()
subdf = df[['Vis_cal', 'PM25', 'RH', 'VC']].dropna()
resampled_df = subdf.resample('3H').mean()


@setFigure(figsize=(8, 6))
def four_quar(subdf):
    item = 'RH'
    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(subdf['PM25'], subdf['Vis_cal'], s=200 * (subdf[item]/ subdf[item].max())**4, c=subdf['VC'], norm=plt.Normalize(vmin=0, vmax=2000), cmap='YlGnBu')
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

    ax.legend(loc='center right', bbox_to_anchor=(0.8, 0.3, 0.2, 0.2), scatterpoints=1, frameon=False, labelspacing=0.5, title=unit(item))

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


if __name__ == '__main__':
    four_quar(resampled_df)