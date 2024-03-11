from DataPlot import *
import pandas as pd
import matplotlib.pyplot as plt

from heatmap import heatmap, corrplot
import seaborn as sns

df = DataBase

# plot.fRH_plot()
# plot.ammonium_rich(df)
# plot.wind_rose(df['WS'], df['WD'])
# plot.MLR_IMPROVE()


# @set_figure
# def heatmap(x, y, size, color):
#     fig, ax = plt.subplots()
#
#     # Mapping from column names to integer coordinates
#     x_labels = [v for v in sorted(x.unique())]
#     y_labels = [v for v in sorted(y.unique())]
#     x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
#     y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
#
#     size_scale = 500
#     ax.scatter(
#         x=x.map(x_to_num),  # Use mapping for x
#         y=y.map(y_to_num),  # Use mapping for y
#         s=size * size_scale,  # Vector of square sizes, proportional to size parameter
#         marker='s'  # Use square as scatterplot marker
#     )
#
#     # Show column labels on the axes
#     ax.set_xticks([x_to_num[v] for v in x_labels])
#     ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
#     ax.set_yticks([y_to_num[v] for v in y_labels])
#     ax.set_yticklabels(y_labels)
#
#     ax.grid(False, 'major')
#     ax.grid(True, 'minor')
#     ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
#     ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
#
#     ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
#     ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
#
#     n_colors = 256  # Use 256 colors for the diverging color palette
#     palette = sns.diverging_palette(20, 220, n=n_colors)  # Create the palette
#     color_min, color_max = [-1, 1]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation
#
#     def value_to_color(val):
#         val_position = float((val - color_min)) / (
#                     color_max - color_min)  # position of value in the input range, relative to the length of the input range
#         ind = int(val_position * (n_colors - 1))  # target index in the color palette
#         return palette[ind]
#
#     ax.scatter(
#         x=x.map(x_to_num),
#         y=y.map(y_to_num),
#         s=size * size_scale,
#         c=color.apply(value_to_color),  # Vector of square color values, mapped to color palette
#         marker='s'
#     )
#
#
# data = DataBase
# columns = ['Extinction', 'Scattering', 'Absorption', 'PM1', 'PM25', 'PM10', 'NO', 'NO2', 'NOx', 'O3', 'Benzene', 'Toluene']
# corr = data[columns].corr()
# corr = pd.melt(corr.reset_index(), id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
# corr.columns = ['x', 'y', 'value']
#
#
# heatmap(
#     x=corr['x'],
#     y=corr['y'],
#     size=corr['value'].abs(),
#     color=corr['value']
# )
