import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal, Optional
from DataPlot.plot.core import *

__all__ = ['barplot_extend',
           'barplot_concen',
           'barplot_combine',
           ]


@set_figure(figsize=(10, 6))
def barplot_extend(data_set: dict, data_std: dict,
                   labels,
                   unit: str = None,
                   orientation: Literal["va", "ha"] = 'va',
                   symbol=True,
                   **kwargs):
    """

    Parameters
    ----------
    data_set : dict
        A mapping from category_names to a list of species mean.
    data_std : dict
        A mapping from category_names to a list of species std.
    labels : list of str
        The species name.
    symbol : bool
        Whether to display values for each species.
    orientation : str
        The orientation of the core.

    Returns
    -------
    matplotlib.Axes

    """
    # data process
    data = np.array(list(data_set.values()))

    if data_std is None:
        data_std = np.zeros(data.shape)
    else:
        data_std = np.array(list(data_std.values()))

    groups, species = data.shape
    groups_arr = np.arange(groups)
    species_arr = np.arange(species)

    # figure info
    category_names = ticks = kwargs.get('ticks') or list(data_set.keys())
    title = kwargs.get('title') or ''
    colors = kwargs.get('colors') or Color.getColor(num=species)

    fig, ax = plt.subplots()

    width = 0.1
    block = width / 4

    for i in range(species):
        val = data[:, i]
        std = (0,) * groups, data_std[:, i]
        if orientation == 'va':
            _ = plt.bar(groups_arr + (i + 1) * (width + block), val, yerr=std, width=width, color=colors[i],
                        edgecolor=None, capsize=None)
        if orientation == 'ha':
            _ = plt.barh(groups_arr + (i + 1) * (width + block), val, xerr=std, height=width, color=colors[i],
                         edgecolor=None, capsize=None)
        if symbol:
            ax.bar_label(_, fmt='%.2f', label_type='center', padding=0, fontsize=8, weight='bold')

    if orientation == 'va':
        plt.xticks(groups_arr + (species / 2 + 0.5) * (width + block), category_names, weight='bold')
        plt.ylabel(Unit(f'{unit}'))
        plt.title(fr'$\bf {title}$')
        plt.legend(labels, loc='best', prop={'size': 12})
        plt.show()
        # fig.savefig(f"IMPR_exd_va_{title}")

    if orientation == 'ha':
        ax.invert_yaxis()
        plt.yticks(groups_arr + 3.5 * (width + block), category_names, weight='bold')
        plt.xlabel(Unit(f'{unit}'))
        plt.title(fr'$\bf {title}$')
        plt.legend(labels, loc='best', prop={'size': 12})
        plt.show()
        # fig.savefig(f"IMPR_exd_ha_{title}")

    return ax


@set_figure(figsize=(10, 6), fs=12)
def barplot_concen(data_set: dict[str, pd.DataFrame],
                   items: list[str],
                   labels: list[str],
                   unit: str,
                   orientation: Literal["va", "ha"] = 'va',
                   symbol=True,
                   **kwargs):
    """

    Parameters
    ----------
    data_set : dict
        A mapping from category_names to a list of species mean.
    labels : list of str
        The species name.
    symbol : bool
        Whether to display values for each species.
    orientation : str
        The orientation of the core.

    Returns
    -------
    matplotlib.Axes

    """
    # data process

    abs_data = np.array([df[items].dropna().mean().values for df in data_set.values()])

    groups, species = abs_data.shape
    groups_arr = np.arange(groups)
    species_arr = np.arange(species)

    total = np.array([abs_data.sum(axis=1), ] * species).T

    data = abs_data / total * 100
    data_cum = data.cumsum(axis=1)

    # figure info
    category_names = kwargs.get('ticks') or list(data_set.keys())
    title = kwargs.get('title') or ''
    title_format = fr'$\bf {title}$'
    colors = kwargs.get('colors') or Color.getColor(num=species)

    fig, ax = plt.subplots()

    for i in range(species):
        widths = data[:, i]
        starts = data_cum[:, i] - data[:, i]

        if orientation == 'va':
            _ = plt.bar(groups_arr, widths, bottom=starts, width=0.7, color=colors[i], label=labels[i],
                        edgecolor=None, capsize=None)
        if orientation == 'ha':
            _ = plt.barh(groups_arr, widths, left=starts, height=0.7, color=colors[i], label=labels[i],
                         edgecolor=None, capsize=None)
        if symbol:
            ax.bar_label(_, fmt='%.2f%%', label_type='center', padding=0, fontsize=10, weight='bold')

    if orientation == 'va':
        ax.set_xticks(groups_arr, category_names, weight='normal')
        ax.legend(labels, ncol=species, bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 12}, frameon=False)
        ax.set_xlabel(Unit(unit))
        ax.set_ylabel(r'$\bf Contribution\ (\%)$')
        # ax.yaxis.set_visible(False)
        ax.set_ylim(0, np.sum(data, axis=1).max())
        # fig.savefig(f"IMPR_con_va_{title}")

    if orientation == 'ha':
        ax.invert_yaxis()

        ax.set_yticks(groups_arr, category_names, weight='bold')
        # ax.legend(labels, ncol=species, bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 12}, frameon=False)
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        # fig.savefig(f"IMPR_con_ha_{title}", transparent=True)

    return ax


@set_figure(figsize=(10, 6), fs=16)
def barplot_combine(data_set, labels, data_ALWC, data_ALWC_std, data_std=None, unit: str = None,
                    orientation: Literal["va", "ha"] = 'va',
                    symbol=True,
                    **kwargs):
    """

    Parameters
    ----------
    data_set : list of dict
        A mapping from category_names to a list of species.
    style : str
        The type of bar chart 'relative' or 'absolute'.
    symbol : bool
        Whether to display values for each species.

    Returns
    -------
    matplotlib.Axes

    """

    # data process
    data = np.array(list(data_set.values()))
    data2 = np.array(list(data_ALWC.values()))
    if data_std is None:
        data_std = np.zeros(data.shape)
    else:
        data_std = np.array(list(data_std.values()))

    data_std2 = np.array(list(data_ALWC_std.values()))

    groups, species = data.shape
    groups_arr = np.arange(groups)
    species_arr = np.arange(species)

    # figure info
    category_names = ticks = kwargs.get('ticks') or list(data_set.keys())
    title = kwargs.get('title') or ''
    colors = kwargs.get('colors') or Color.getColor(num=species)

    fig, ax = plt.subplots()

    width = 0.1
    block = width / 4
    axx = []
    for i in range(species):
        val = data[:, i]
        std = (0,) * groups, data_std[:, i]

        _ = plt.bar(groups_arr + (i + 1) * (width + block), val, yerr=std, width=width, color=colors[i],
                    edgecolor=None, capsize=None)
        axx.append(_)
        if symbol:
            ax.bar_label(_, fmt='%.2f', label_type='center', padding=0, fontsize=8, weight='bold')

        if i == 0:
            val2 = data2[:, 0]
            std2 = (0,) * groups, data_std2[:, 0]
            _ = plt.bar(groups_arr + (i + 1) * (width + block), val2, yerr=std2, bottom=val, width=width,
                        color=colors[-1],
                        edgecolor=None, capsize=None)
            if symbol:
                ax.bar_label(_, fmt='%.2f', label_type='center', padding=0, fontsize=8, weight='bold')

        if i == 1:
            val2 = data2[:, 1]
            std2 = (0,) * groups, data_std2[:, 1]
            _ = plt.bar(groups_arr + (i + 1) * (width + block), val2, yerr=std2, bottom=val, width=width,
                        color=colors[-1],
                        edgecolor=None, capsize=None)
            if symbol:
                ax.bar_label(_, fmt='%.2f', label_type='center', padding=0, fontsize=8, weight='bold')

        if i == 4:
            val2 = data2[:, 2]
            std2 = (0,) * groups, data_std2[:, 2]
            __ = plt.bar(groups_arr + (i + 1) * (width + block), val2, yerr=std2, bottom=val, width=width,
                         color=colors[-1],
                         edgecolor=None, capsize=None)
            if symbol:
                ax.bar_label(__, fmt='%.2f', label_type='center', padding=0, fontsize=8, weight='bold')
        if i == 5:
            axx.append(__)

    plt.xticks(groups_arr + (species / 2 + 0.5) * (width + block), category_names, weight='bold')
    plt.ylabel(Unit(f'{unit}'))
    plt.title(fr'$\bf {title}$')
    plt.legend(axx, labels, loc='best', prop={'size': 12}, frameon=False)
    plt.show()
    # fig.savefig(f"IMPR_exd_va_{title}")

    return ax
