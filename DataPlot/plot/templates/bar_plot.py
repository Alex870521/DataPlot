import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal
from DataPlot.plot.core import *

__all__ = ['barplot',
           ]


def auto_label(pct):
    if pct > 3:
        return '{:.2f}'.format(pct)
    else:
        return ''


@set_figure(figsize=(8, 5), fs=12)
def barplot(data_set: pd.DataFrame | dict,
            data_std: pd.DataFrame | None,
            labels: list[str],
            unit: str,
            display: Literal["stacked", "dispersed"] = "dispersed",
            orientation: Literal["va", "ha"] = 'va',
            ax: plt.Axes | None = None,
            symbol=True,
            **kwargs):
    """
    Parameters
    ----------
    data_set : pd.DataFrame or dict
        A mapping from category names to a list of species mean or a DataFrame with columns as categories and values as means.
    data_std : pd.DataFrame or None
        A DataFrame with standard deviations corresponding to data_set, or None if standard deviations are not provided.
    labels : list of str
        The species names.
    unit : str
        The unit for the values.
    display : {'stacked', 'dispersed'}, default 'dispersed'
        Whether to display the bars stacked or dispersed.
    orientation : {'va', 'ha'}, default 'va'
        The orientation of the bars, 'va' for vertical and 'ha' for horizontal.
    ax : plt.Axes or None, default None
        The Axes object to plot on. If None, a new figure and Axes are created.
    symbol : bool, default True
        Whether to display values for each bar.
    kwargs : dict
        Additional keyword arguments passed to the barplot function.

    Returns
    -------
    matplotlib.Axes
        The Axes object containing the plot.

    """
    # data process
    data = data_set.values

    if data_std is None:
        data_std = np.zeros(data.shape)
    else:
        data_std = data_std.values

    groups, species = data.shape
    groups_arr = np.arange(groups)
    species_arr = np.arange(species)

    total = np.array([data.sum(axis=1), ] * species).T

    pct_data = data / total * 100
    data_cum = pct_data.cumsum(axis=1)

    # figure info
    category_names = kwargs.get('ticks') or list(data_set.index)
    title = kwargs.get('title', '')
    colors = kwargs.get('colors') or (Color.colors1 if species == 6 else Color.getColor(num=species))

    if ax is None:
        fig, ax = plt.subplots()

    if display == "stacked":
        for i in range(species):
            widths = pct_data[:, i]
            starts = data_cum[:, i] - pct_data[:, i]

            if orientation == 'va':
                _ = plt.bar(groups_arr, widths, bottom=starts, width=0.7, color=colors[i], label=labels[i],
                            edgecolor=None, capsize=None)
            if orientation == 'ha':
                _ = plt.barh(groups_arr, widths, left=starts, height=0.7, color=colors[i], label=labels[i],
                             edgecolor=None, capsize=None)
            if symbol:
                ax.bar_label(_, fmt=auto_label, label_type='center', padding=0, fontsize=10, weight='bold')

    if display == "dispersed":
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
                ax.bar_label(_, fmt=auto_label, label_type='center', padding=0, fontsize=8, weight='bold')

    if orientation == 'va':
        xticks = groups_arr + (species / 2 + 0.5) * (width + block) if display == "dispersed" else groups_arr
        ax.set_xticks(xticks, category_names, weight='bold')
        ax.set_ylabel(Unit(unit) if display == "dispersed" else r'$\bf Contribution\ (\%)$')
        ax.set_ylim(0, None if display == "dispersed" else 100)
        ax.legend(labels, bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 12})

    if orientation == 'ha':
        ax.invert_yaxis()
        yticks = groups_arr + 3.5 * (width + block) if display == "dispersed" else groups_arr
        ax.set_yticks(yticks, category_names, weight='bold')
        ax.set_xlabel(Unit(unit) if display == "dispersed" else r'$\bf Contribution\ (\%)$')
        ax.set_xlim(0, None if display == "dispersed" else 100)
        ax.legend(labels, bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 12})

    # fig.savefig(f"Barplot_{title}")

    return ax
