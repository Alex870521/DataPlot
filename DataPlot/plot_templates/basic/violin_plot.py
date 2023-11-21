import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot_templates import set_figure, unit, getColor


@set_figure(figsize=(6, 6))
def violin(means, data_set, **kwargs):  # Distribution pattern of light extinction
    grps = len(means)

    fig, ax = plt.subplots(1, 1)
    width = 0.6
    block = width / 2
    x_position = np.arange(grps)

    plt.boxplot(data_set, positions=x_position, widths=0.15,
                showfliers=False, showmeans=True, meanline=False, patch_artist=True,
                capprops    =dict(linewidth=0),
                whiskerprops=dict(linewidth=1.5, color='k', alpha=1),
                boxprops    =dict(linewidth=1.5, color='k', facecolor='#4778D3', alpha=1),
                meanprops   =dict(marker='o', markeredgecolor='black', markerfacecolor='white', markersize=6),
                medianprops =dict(linewidth=1.5, ls='-', color='k', alpha=1))

    violin = sns.violinplot(data=data_set, scale='area', color='#4778D3', inner=None)
    for violin, alpha in zip(ax.collections[:], [0.5] * len(ax.collections[:])):
        violin.set_alpha(alpha)
        violin.set_edgecolor(None)

    plt.scatter(x_position, means, marker='o', facecolor='white', edgecolor='k', s=10)

    xlim = kwargs.get('xlim') or (x_position[0]-(width/2+block), x_position[-1]+(width/2+block))
    ylim = kwargs.get('ylim') or (0, 500)
    xlabel = kwargs.get('xlabel') or ''
    ylabel = kwargs.get('ylabel') or unit('Extinction')
    ticks = kwargs.get('ticks') or np.arange(grps)
    title = kwargs.get('title') or ''
    title = title.replace(' ', '\ ')
    title_format = fr'$\bf {title}$'

    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(x_position, ticks, fontweight='bold', fontsize=12)
    ax.set_title(title_format)

    plt.show()
    fig.savefig(f'Violin')