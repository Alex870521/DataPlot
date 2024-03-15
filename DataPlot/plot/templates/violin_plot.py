import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional
from DataPlot.plot.core import *


@set_figure(figsize=(6, 6), fs=12)
def violin(data_set: dict[str, pd.DataFrame],
           unit: str,
           ax: Optional[plt.Axes] = None,
           **kwargs):

    grps = len(data_set)

    if ax is None:
        fig, ax = plt.subplots()

    width = 0.6
    block = width / 2
    x_position = np.arange(grps)

    data = [df[unit].dropna().values for df in data_set.values()]

    plt.boxplot(data, positions=x_position, widths=0.15,
                showfliers=False, showmeans=True, meanline=False, patch_artist=True,
                capprops=dict(linewidth=0),
                whiskerprops=dict(linewidth=1.5, color='k', alpha=1),
                boxprops=dict(linewidth=1.5, color='k', facecolor='#4778D3', alpha=1),
                meanprops=dict(marker='o', markeredgecolor='black', markerfacecolor='white', markersize=6),
                medianprops=dict(linewidth=1.5, ls='-', color='k', alpha=1))

    sns.violinplot(data=data, density_norm='area', color='#4778D3', inner=None)

    for violin, alpha in zip(ax.collections[:], [0.5] * len(ax.collections[:])):
        violin.set_alpha(alpha)
        violin.set_edgecolor(None)

    plt.scatter(x_position, [df[unit].dropna().values.mean() for df in data_set.values()], marker='o',
                facecolor='white', edgecolor='k', s=10)

    xlim = kwargs.get('xlim') or (x_position[0] - (width / 2 + block), x_position[-1] + (width / 2 + block))
    ylim = kwargs.get('ylim') or (0, None)
    xlabel = kwargs.get('xlabel') or ''
    ylabel = kwargs.get('ylabel') or Unit(unit)
    xticks = kwargs.get('xticks') or [x.replace('-', '\n') for x in list(data_set.keys())]
    title = kwargs.get('title') or ''

    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_xticks(x_position, xticks, fontweight='bold', fontsize=12)

    # fig.savefig(f'Violin_{unit}')
    plt.show()
