import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from .ColorHandler import getColor
from .UnitHandler import unit

__all__ = ['set_figure',
           'prop_text',
           'prop_legend']


font_scalings = {
    'xx-small': 0.579,
    'x-small': 0.694,
    'small': 0.833,
    'medium': 1.0,
    'large': 1.200,
    'x-large': 1.440,
    'xx-large': 1.728,
    'larger': 1.2,
    'smaller': 0.833
}

prop_legend = {'size': 14, 'family': 'Times New Roman', 'weight': 'bold'}
prop_text = {'fontsize': 14, 'fontfamily': 'Times New Roman', 'fontweight': 'bold'}


def set_figure(func=None, *, figsize=None, fs=None, fw=None, titlesize=None):
    def decorator(_func):
        @wraps(_func)
        def wrapper(*args, **kwargs):
            plt.rcParams['mathtext.fontset'] = 'custom'
            plt.rcParams['mathtext.rm'] = 'Times New Roman'
            plt.rcParams['mathtext.it'] = 'Times New Roman: italic'
            plt.rcParams['mathtext.bf'] = 'Times New Roman: bold'
            plt.rcParams['mathtext.default'] = 'regular'

            # The font properties used by `text.Text`.
            # The text, annotate, label, title, ticks, are used to create text
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.weight'] = 'normal'
            plt.rcParams['font.size'] = fs or 16

            plt.rcParams['axes.titlelocation'] = 'center'
            plt.rcParams['axes.titleweight'] = 'bold'
            plt.rcParams['axes.titlesize'] = titlesize or 20
            plt.rcParams['axes.labelweight'] = 'bold'

            plt.rcParams['xtick.labelsize'] = 'medium'
            plt.rcParams['ytick.labelsize'] = 'medium'

            # matplotlib.font_manager.FontProperties ---> matplotlib.rcParams
            plt.rcParams['legend.loc'] = 'best'
            plt.rcParams['legend.frameon'] = False
            plt.rcParams['legend.fontsize'] = 'medium'
            # plt.rcParams['legend.fontweight'] = 'bold'  #key error
            plt.rcParams['legend.handlelength'] = 1.5

            plt.rcParams['figure.figsize'] = figsize or (8, 8)
            plt.rcParams['figure.constrained_layout.use'] = True
            plt.rcParams['figure.dpi'] = 150

            plt.rcParams['savefig.transparent'] = True

            result = _func(*args, **kwargs)

            return result
        return wrapper

    if func is None:
        return decorator

    else:
        return decorator(func)


if __name__ == '__main__':

    @set_figure(figsize=(6, 6), fw=16, fs=16)
    def test(**kwargs):
        data = np.array([1.01, 2.20, 1.47, 1.88, 1.42, 3.78])
        fig, ax = plt.subplots(1, 1)
        col = ['#FF3333', '#33FF33', '#FFFF33', '#5555FF', '#B94FFF', '#FFA500']
        x_position = np.array([0, 1, 2, 3, 4, 5])
        plt.bar(x_position, data, color=getColor(6))

        xlim = kwargs.get('xlim') or (-0.5, 5.5)
        ylim = kwargs.get('ylim') or (1, 4)
        xlabel = kwargs.get('xlabel') or unit.Factor
        ylabel = kwargs.get('ylabel') or unit.fRH
        title = kwargs.get('title') or ''

        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks(x_position, labels=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], )
        ax.set_yticks(ax.get_yticks())
        ax.set_title(title)
        ax.legend(prop=dict(weight='bold', size=12))

    test(title='Test the title')

