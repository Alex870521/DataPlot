from os.path import join as pth
import matplotlib.pyplot as plt
import numpy as np
from DataPlot.plot_templates import set_figure, unit, getColor, linecolor

prop_text = {'fontsize': 14, }
prop_legend = {'fontsize': 14, }

__all__ = ['pie_mass', 'pie_ext', 'donuts_mass', 'donuts_ext']


def inner_pct(pct, symbol=True):
    if symbol:
        if pct < 8:
            return ''
        else:
            return '{:.1f}%'.format(pct)
    else:
        return ''


def outer_pct(pct, symbol=True):
    if symbol:
        if pct > 8:
            return ''
        else:
            return '{:.1f}%'.format(pct)
    else:
        return ''


@set_figure
def pie_mass(data_set, labels, style='pie', title='', symbol=True):
    """

    Parameters
    ----------
    data_set : dict
        A mapping from category_names to a list of species.
    labels : list of str
        The category labels.
    style : str
        The type of pie chart 'pie' or 'donut'.
    title : str

    symbol : bool
        Whether to display values for each species.

    Returns
    -------
    (object, object)
        matplotlib.figure, matplotlib.axes

    """
    category_names = np.array(list(data_set.keys()))
    data = np.array(list(data_set.values()))

    pies, species = data.shape
    label_colors = getColor(kinds='1') if species == 6 else getColor(kinds='2')

    radius = 4
    width = 4 if style == 'pie' else 1
    text = [''] * pies if style == 'pie' else [r'$\bf Total\ PM_{2.5}$' + '\n\n' + '{:.2f}'.format(x) + r'$\bf\ (\mu g/m^3)$' for x in data.sum(axis=1)]
    pct_distance = 0.6 if style == 'pie' else 0.88

    fig, ax = plt.subplots(1, pies, figsize=(pies*3, 3), dpi=150, constrained_layout=True)

    for i in range(pies):
        ax[i].pie(data[i], labels=None, colors=label_colors, textprops=prop_text,
                  autopct=lambda pct: inner_pct(pct, symbol=symbol),
                  pctdistance=pct_distance, radius=radius, wedgeprops=dict(width=width, edgecolor='w'))

        ax[i].pie(data[i], labels=None, colors=label_colors, textprops=prop_text,
                  autopct=lambda pct: outer_pct(pct, symbol=symbol),
                  pctdistance=1.2, radius=radius, wedgeprops=dict(width=width, edgecolor='w'))

        ax[i].axis('equal')
        ax[i].text(0, 0, text[i], fontdict=prop_text, ha='center', va='center')
        ax[i].set_title(rf'$\bf {category_names[i]}$')

    plt.show()
    fig.savefig(pth(f"IMPROVE_mass_{style}_{title}"))
    return fig, ax


@set_figure
def pie_ext(data_set, labels, style='pie', title='', symbol=True):
    """

    Parameters
    ----------
    data_set : dict
        A mapping from category_names to a list of species.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *labels*.
    labels : list of str
        The category labels.
    style : str
        The type of pie chart 'pie' or 'donut'.
    title : str
        default: ''
    symbol : bool
        Whether to display values for each species.

    """
    category_names = list(data_set.keys())
    data = np.array(list(data_set.values()))

    pies, species = data.shape
    label_colors = getColor(kinds='3') if species == 6 else getColor(kinds='4')

    radius = 4
    width = 4 if style == 'pie' else 1
    text = [''] * pies if style == 'pie' else [(r'$\bf Total\ Extinction$' + '\n\n' + '{:.2f}' + r'$\bf \ (1/Mm)$').format(x) for x in data.sum(axis=1)]
    pct_distance = 0.6 if style == 'pie' else 0.88

    fig, ax = plt.subplots(1, pies, figsize=(pies*3, 3), dpi=150)

    for i in range(pies):
        ax[i].pie(data[i], labels=None, colors=label_colors, textprops=prop_text,
                  autopct=lambda pct: inner_pct(pct, symbol=symbol),
                  pctdistance=pct_distance, radius=radius, wedgeprops=dict(width=width, edgecolor='w'))

        ax[i].pie(data[i], labels=None, colors=label_colors, textprops=prop_text,
                  autopct=lambda pct: outer_pct(pct, symbol=symbol),
                  pctdistance=1.3, radius=radius, wedgeprops=dict(width=width, edgecolor='w'))
        ax[i].axis('equal')
        ax[i].text(0, 0, text[i], fontdict=prop_text, ha='center', va='center')
        ax[i].set_title(rf'$\bf {category_names[i]}$', pad=-10)

    plt.show()
    fig.savefig(pth(f"IMPROVE_ext_{style}_{title}"), transparent=True)
    return fig, ax


@set_figure(figsize=(10, 6))
def donuts_mass(data_set, labels, style='donut', title='', symbol=True):
    """
    Parameters
    ----------
    data_set : dict
        A mapping from category_names to a list of species.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *labels*.
    labels : list of str
        The category labels.
    style : str
        The type of pie chart is 'donuts'.
    title : str
        default: ''
    symbol : bool
        Whether to display values for each species.
    """
    labels = 'AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'Others'

    values1 = np.array(list(data_set.values()))[3]
    values2 = np.array(list(data_set.values()))[2]
    values3 = np.array(list(data_set.values()))[1]

    colors1 = getColor(kinds='3-2')

    def adjust_opacity(color, alpha):
        # 將顏色轉換為RGB表示
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        # 調整透明度
        r_new = int(alpha * r + (1 - alpha) * 255)
        g_new = int(alpha * g + (1 - alpha) * 255)
        b_new = int(alpha * b + (1 - alpha) * 255)
        # 轉換為新的色碼
        new_color = '#{:02X}{:02X}{:02X}'.format(r_new, g_new, b_new)
        return new_color

    colors2 = [adjust_opacity(color, 0.8) for color in colors1]
    colors3 = [adjust_opacity(color, 0.6) for color in colors1]

    fig, ax = plt.subplots(1, 1)
    ax.pie(values1, labels=None, colors=colors1, textprops=prop_text,
           autopct='%1.1f%%',
           pctdistance=0.9, radius=14, wedgeprops=dict(width=3, edgecolor='w'))

    ax.pie(values2, labels=None, colors=colors2, textprops=prop_text,
           autopct='%1.1f%%',
           pctdistance=0.85, radius=11, wedgeprops=dict(width=3, edgecolor='w'))

    ax.pie(values3, labels=None, colors=colors3, textprops=prop_text,
           autopct='%1.1f%%',
           pctdistance=0.80, radius=8, wedgeprops=dict(width=3, edgecolor='w'))

    text = r'$\bf Average\ (\mu g/m^3)$' + '\n\n' + 'Event : ' + "{:.2f}".format(np.sum(values1)) + '\n' + \
           'Transition : ' + "{:.2f}".format(np.sum(values2)) + '\n' + \
           'Clean : ' + "{:.2f}".format(np.sum(values3))

    ax.text(0, 0, text, fontdict=prop_text, ha='center', va='center')
    ax.axis('equal')
    ax.set_title(f'{title}', size=20, weight='bold')

    ax.legend(labels, loc='center', prop=prop_legend, title_fontproperties=dict(weight='bold'),
              title='Outer : Event' + '\n' + 'Middle : Transition' + '\n' + 'Inner : Clean',
              bbox_to_anchor=(0.66, 0, 0.5, 1), frameon=False)
    plt.show()
    fig.savefig(pth(f"IMPROVE_mass_donuts_{title}"), transparent=True)


@set_figure(figsize=(10, 6))
def donuts_ext(data_set, labels, style='donut', title='', symbol=True):
    """
    Parameters
    ----------
    data_set : dict
        A mapping from category_names to a list of species.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *labels*.
    labels : list of str
        The category labels.
    style : str
        The type of pie chart is 'donuts'.
    title : str
        default: ''
    symbol : bool
        Whether to display values for each species.
    """
    labels = labels

    values1 = np.array(list(data_set.values()))[3]
    values2 = np.array(list(data_set.values()))[2]
    values3 = np.array(list(data_set.values()))[1]

    colors1 = getColor(kinds='3')
    if len(labels) == 9:
        colors1 = getColor(kinds='4-1')

    def adjust_opacity(color, alpha):
        # 將顏色轉換為RGB表示
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        # 調整透明度
        r_new = int(alpha * r + (1 - alpha) * 255)
        g_new = int(alpha * g + (1 - alpha) * 255)
        b_new = int(alpha * b + (1 - alpha) * 255)
        # 轉換為新的色碼
        new_color = '#{:02X}{:02X}{:02X}'.format(r_new, g_new, b_new)
        return new_color

    colors2 = [adjust_opacity(color, 0.8) for color in colors1]
    colors3 = [adjust_opacity(color, 0.6) for color in colors1]

    fig, ax = plt.subplots(1, 1)
    ax.pie(values1, labels=None, colors=colors1, textprops=prop_text,
           autopct='%1.1f%%',
           pctdistance=0.9, radius=14, wedgeprops=dict(width=3, edgecolor='w'))

    ax.pie(values2, labels=None, colors=colors2, textprops=prop_text,
           autopct='%1.1f%%',
           pctdistance=0.85, radius=11, wedgeprops=dict(width=3, edgecolor='w'))

    ax.pie(values3, labels=None, colors=colors3, textprops=prop_text,
           autopct='%1.1f%%',
           pctdistance=0.80, radius=8, wedgeprops=dict(width=3, edgecolor='w'))

    text = 'Average (1/Mm)' + '\n\n' + 'Event : ' + "{:.2f}".format(np.sum(values1)) + '\n' + \
           'Transition : ' + "{:.2f}".format(np.sum(values2)) + '\n' + \
           'Clean : ' + "{:.2f}".format(np.sum(values3))

    ax.text(0, 0, text, fontdict=prop_text, ha='center', va='center')
    ax.axis('equal')
    ax.set_title(f'{title}', size=20, weight='bold')

    ax.legend(labels, loc='center', prop=prop_legend, title_fontproperties=dict(weight='bold'),
              title='Outer : Event' + '\n' + 'Middle : Transition' + '\n' + 'Inner : Clean',
              bbox_to_anchor=(0.66, 0, 0.5, 1), frameon=False)
    plt.show()
    fig.savefig(pth(f"IMPROVE_ext_donuts_{title}"), transparent=True)

