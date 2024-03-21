import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot import *
from scipy.stats import pearsonr

df = DataBase


# Correlation Matrix
@set_figure(figsize=(6, 6), fs=8)
def heatmap(df, cmap="RdBu"):
    _corr = df.corr()
    corr = pd.melt(_corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']

    p_values = _corr.apply(lambda col1: _corr.apply(lambda col2: pearsonr(col1, col2)[1]))
    p_values = p_values.mask(p_values > 0.05)
    p_values = pd.melt(p_values.reset_index(), id_vars='index').dropna()
    p_values.columns = ['x', 'y', 'value']

    fig, ax = plt.subplots()

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(corr['x'].unique())]
    y_labels = [v for v in sorted(corr['y'].unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=90, horizontalalignment='center')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    # ax.tick_params(axis='both', which='major', direction='out', top=True, left=True)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    n_colors = 256  # Use 256 colors for the diverging color palette
    palette = sns.color_palette(cmap, n_colors=n_colors)  # Create the palette

    # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    point = ax.scatter(
        x=corr['x'].map(x_to_num),
        y=corr['y'].map(y_to_num),
        s=corr['value'].abs() * 70,
        c=corr['value'].apply(value_to_color),  # Vector of square color values, mapped to color palette
        marker='s',
        label='$R^{2}$'
    )

    axes_image = plt.cm.ScalarMappable(cmap=matplotlib.colormaps[cmap])

    cax = inset_axes(ax, width="5%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    plt.subplots_adjust(bottom=0.15, right=0.8)
    cbar = plt.colorbar(mappable=axes_image, cax=cax, label=r'$coefficient\ of\ determinationr\ R^{2}$', )

    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(np.linspace(-1, 1, 5))

    point2 = ax.scatter(
        x=p_values['x'].map(x_to_num),  # Use mapping for x
        y=p_values['y'].map(y_to_num),  # Use mapping for y
        s=10,  # Vector of square sizes, proportional to size parameter
        marker='*',  # Use square as scatterplot marker
        color='k',
        label='p < 0.05'
    )

    # ax.scatter(
    #     x=p_values['x'].map(x_to_num),  # Use mapping for x
    #     y=p_values['y'].map(y_to_num),  # Use mapping for y
    #     s=250,  # Vector of square sizes, proportional to size parameter
    #     marker='s',  # Use square as scatterplot marker
    #     color='k',
    #     facecolors='none', edgecolors='k', linewidth=1.5)

    ax.legend(handles=[point2], labels=['p < 0.05'], bbox_to_anchor=(0.05, 1, 0.1, 0.05))
    plt.show()


if __name__ == '__main__':
    data = DataBase

    columns = ['Extinction', 'Scattering', 'Absorption', 'PM1', 'PM25', 'PM10', 'PBLH', 'VC',
               'AT', 'RH', 'WS', 'NO', 'NO2', 'NOx', 'O3', 'Benzene', 'Toluene',
               'SO2', 'CO', 'THC', 'CH4', 'NMHC', 'NH3', 'HCl', 'HNO2', 'HNO3',
               'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+', 'Cl-', 'NO2-', 'NO3-', 'SO42-',]

    heatmap(data[columns])
