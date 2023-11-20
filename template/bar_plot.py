import matplotlib.pyplot as plt
import numpy as np
from template import set_figure, unit, getColor

__all__ = ['barplot_extend', 'barplot_concen', 'barplot_combine']


def autolabel(bars, x, y, val, threshold=3, symbol=True):
    if symbol:
        for j, bar in enumerate(bars):
            position = x[j]
            height = y[j]
            if bar > threshold:  # 值太小不顯示
                plt.text(position, height, '%s' % (val[j]), fontsize=10,
                         fontname='Times New Roman', weight='bold', ha='center', va='center')
    else:
        return ''


def fmt(_):
    if _ > 3:
        return '%.2f'
    elif _ <= 3:
        return ''


@set_figure(figsize=(10, 6))
def barplot_extend(data_set, labels, data_std='None', symbol=True, orientation='va', **kwargs):
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
        The orientation of the config.

    Returns
    -------
    (object, object)
        matplotlib.figure, matplotlib.axes

    """
    # data processing
    data = np.array(list(data_set.values()))

    if data_std == 'None':
        data_std = np.zeros(data.shape)
    else:
        data_std = np.array(list(data_std.values()))

    groups, species = data.shape
    groups_arr = np.arange(groups)
    species_arr = np.arange(species)

    # figure info
    category_names = ticks = kwargs.get('ticks') or list(data_set.keys())
    title = kwargs.get('title') or ''
    colors = kwargs.get('colors') or plt.colormaps['jet_r'](np.linspace(0.1, 0.9, species))

    fig, ax = plt.subplots(1, 1)

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
        plt.ylabel(unit('Extinction'))
        plt.title(fr'$\bf {title}$')
        plt.legend(labels, loc='upper right', prop={'size': 12}, frameon=False)
        plt.show()
        fig.savefig(f"IMPR_exd_va_{title}")

    if orientation == 'ha':
        ax.invert_yaxis()
        plt.yticks(groups_arr + 3.5 * (width + block), category_names, weight='bold')
        plt.xlabel(unit('Extinction'))
        plt.title(fr'$\bf {title}$')
        plt.legend(labels, loc='upper right', prop={'size': 12}, frameon=False)
        plt.show()
        fig.savefig(f"IMPR_exd_ha_{title}")

    return fig, ax


@set_figure(figsize=(8, 6), fs=16)
def barplot_concen(data_set, labels, symbol=True, orientation='va', figsize=None, **kwargs):
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
        The orientation of the config.

    Returns
    -------
    (object, object)
        matplotlib.figure, matplotlib.axes

    """
    # data processing

    abs_data = np.array(list(data_set.values()))

    groups, species = abs_data.shape
    groups_arr = np.arange(groups)
    species_arr = np.arange(species)

    total = np.array([abs_data.sum(axis=1), ] * species).T

    data = abs_data / total * 100
    data_cum = data.cumsum(axis=1)

    # figure info
    category_names = kwargs.get('ticks') or list(data_set.keys())
    title = kwargs.get('title') or ''
    title = title.replace(' ', '\ ')
    title_format = fr'$\bf {title}$'
    colors = kwargs.get('colors') or plt.colormaps['Blues'](np.linspace(0.2, 0.8, species))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

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
            ax.bar_label(_, fmt='%.2f%%', label_type='center', padding=0, fontsize=12, weight='bold')

    if orientation == 'va':
        # ax.set_xticks(groups_arr, ['Total', '2020 \n Summer  ', '2020 \n Autumn  ', '2020 \n Winter  ', '2021 \n Spring  '], weight='bold', fontsize=12)
        ax.set_xticks(groups_arr, category_names, weight='normal')
        ax.legend(labels, ncol=species, bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 12}, frameon=False)
        ax.set_xlabel(r'$\bf Total\ Light\ Extinction\ (1/Mm)$')
        ax.set_ylabel(r'$\bf Contribution\ (\%)$')
        # ax.yaxis.set_visible(False)
        ax.set_ylim(0, np.sum(data, axis=1).max())
        fig.savefig(f"IMPR_con_va_{title}")

    if orientation == 'ha':
        ax.invert_yaxis()

        ax.set_yticks(groups_arr, category_names, weight='bold')
        # ax.legend(labels, ncol=species, bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 12}, frameon=False)
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        fig.savefig(f"IMPR_con_ha_{title}", transparent=True)
    # plt.axis('off')
    return fig, ax


@set_figure(figsize=(12, 12), fw=16, fs=16)
def barplot_combine(data_set, labels, data_ALWC, data_ALWC_std, data_std='None', symbol=True, orientation='va',
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
    (object, object)
        matplotlib.figure, matplotlib.axes

    """

    # data processing
    data = np.array(list(data_set.values()))
    data2 = np.array(list(data_ALWC.values()))
    if data_std == 'None':
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
    colors = kwargs.get('colors') or plt.colormaps['jet_r'](np.linspace(0.1, 0.9, species))

    fig, ax = plt.subplots(1, 1)

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
    plt.ylabel(unit('Extinction'))
    plt.title(fr'$\bf {title}$')
    plt.legend(axx, labels, loc='upper right', prop={'size': 12}, frameon=False)
    plt.show()
    fig.savefig(f"IMPR_exd_va_{title}")

    return fig, ax


if __name__ == '__main__':
    amb_data = {'Clean': [0.0052, 0.7602, 0.2346],
                'Transition': [0.0044, 0.8555, 0.1401],
                'Event': [0.0025, 0.8916, 0.1059]}
    dry_data = {'Clean': [0.0098, 0.8135, 0.1767],
                'Transition': [0.008, 0.8811, 0.111],
                'Event': [0.0061, 0.9287, 0.0652]}
    barplot_concen(dry_data,
                   labels=[r'$\bf ultrafine$', r'$\bf accumulation$', r'$\bf coarse$'],
                   figsize=(5, 6))