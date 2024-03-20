import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
# from heatmap import heatmap, corrplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot import *


df = DataBase


@set_figure(figsize=(5, 5), fs=8)
def heatmap(x, y, size, color):
    fig, ax = plt.subplots()

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    # ax.scatter(
    #     x=x.map(x_to_num),  # Use mapping for x
    #     y=y.map(y_to_num),  # Use mapping for y
    #     s=size * size_scale,  # Vector of square sizes, proportional to size parameter
    #     marker='s'  # Use square as scatterplot marker
    # )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=90, horizontalalignment='right')
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
    palette = sns.color_palette("seismic_r", n_colors=n_colors)  # Create the palette
    # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        val_position = float((val - color_min)) / (
                color_max - color_min)  # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * 300,
        c=color.apply(value_to_color),  # Vector of square color values, mapped to color palette
        marker='s'
    )

    axes_image = plt.cm.ScalarMappable(cmap=matplotlib.colormaps['seismic_r'])

    cax = inset_axes(ax, width="5%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    plt.subplots_adjust(right=0.8)
    cbar = plt.colorbar(mappable=axes_image, cax=cax, label=r'$R^{2}$',)

    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(np.linspace(-1, 1, 5))

    plt.show()


if __name__ == '__main__':
    data = DataBase
    columns = ['Extinction', 'Scattering', 'Absorption', 'PM1', 'PM25', 'PM10', 'NO', 'NO2', 'NOx', 'O3', 'Benzene',
               'Toluene']
    corr = data[columns].corr()
    corr = pd.melt(corr.reset_index(), id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']

    heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs(),
        color=corr['value']
    )
