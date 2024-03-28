import matplotlib.pyplot as plt
from functools import wraps
from tqdm import tqdm


__all__ = ['set_figure']


# For more details please see https://matplotlib.org/stable/users/explain/customizing.html


def set_figure(func=None,
               *,
               figsize: tuple | None = None,
               fs: int | None = None,
               fw: str = None,
               ):
    def decorator(_func):
        @wraps(_func)
        def wrapper(*args, **kwargs):
            with tqdm(total=1, desc=f"Plot: {_func.__name__}", bar_format="{l_bar}{bar}|") as progress:
                plt.rcParams['mathtext.fontset'] = 'custom'
                plt.rcParams['mathtext.rm'] = 'Times New Roman'
                plt.rcParams['mathtext.it'] = 'Times New Roman: italic'
                plt.rcParams['mathtext.bf'] = 'Times New Roman: bold'
                plt.rcParams['mathtext.default'] = 'regular'

                # The font properties used by `text.Text`.
                # The text, annotate, label, title, ticks, are used to create text
                plt.rcParams['font.family'] = 'Times New Roman'
                plt.rcParams['font.weight'] = fw or 'normal'
                plt.rcParams['font.size'] = fs or 12

                plt.rcParams['axes.titlesize'] = 'large'
                plt.rcParams['axes.titleweight'] = 'bold'
                plt.rcParams['axes.labelweight'] = 'bold'

                plt.rcParams['xtick.labelsize'] = 'medium'
                plt.rcParams['ytick.labelsize'] = 'medium'

                # matplotlib.font_manager.FontProperties ---> matplotlib.rcParams
                plt.rcParams['legend.loc'] = 'best'
                plt.rcParams['legend.frameon'] = False
                plt.rcParams['legend.fontsize'] = 'small'
                plt.rcParams['legend.title_fontsize'] = 'medium'
                plt.rcParams['legend.handlelength'] = 1.5

                plt.rcParams['figure.figsize'] = figsize or (6, 5)
                plt.rcParams['figure.dpi'] = 150
                plt.rcParams['figure.subplot.left'] = 0.125
                plt.rcParams['figure.subplot.right'] = 0.85
                plt.rcParams['figure.subplot.bottom'] = 0.125
                plt.rcParams['figure.subplot.top'] = 0.88
                # plt.rcParams['figure.autolayout'] = True
                # plt.rcParams['figure.constrained_layout.use'] = True

                plt.rcParams['savefig.transparent'] = True

                result = _func(*args, **kwargs)

                progress.update()

            return result

        return wrapper

    if func is None:
        return decorator

    else:
        return decorator(func)
