import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot import set_figure, unit
from DataPlot.process import DataBase

df = DataBase
subdf = df[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna()
resampled_df = subdf.resample('3h').mean()


@set_figure(figsize=(8, 6))
def four_quar(subdf):
    item = 'RH'
    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(subdf['PM25'], subdf['Vis_LPV'], s=50 * (subdf[item] / subdf[item].max())**4, c=subdf['VC'], norm=plt.Normalize(vmin=0, vmax=2000), cmap='YlGnBu')
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

    plt.show()


if __name__ == '__main__':
    four_quar(resampled_df)
