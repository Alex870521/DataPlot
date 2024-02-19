import matplotlib.pyplot as plt
import time
from functools import wraps
from typing import Optional


# For more details please seehttps://matplotlib.org/stable/users/explain/customizing.html

# font_scalings = {
#     'xx-small': 0.579,
#     'x-small': 0.694,
#     'small': 0.833,
#     'medium': 1.0,
#     'large': 1.200,
#     'x-large': 1.440,
#     'xx-large': 1.728,
#     'larger': 1.2,
#     'smaller': 0.833
# }


def set_figure(func=None,
               *,
               figsize: Optional[tuple] = None,
               titlesize: Optional[str] = None,
               fs: Optional[str] = None,
               fw: Optional[str] = None,
               ):

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
            plt.rcParams['font.weight'] = fw or 'normal'
            plt.rcParams['font.size'] = fs or 14

            plt.rcParams['axes.titlelocation'] = 'center'
            plt.rcParams['axes.titleweight'] = 'bold'
            plt.rcParams['axes.titlesize'] = 'large'
            plt.rcParams['axes.labelweight'] = 'bold'

            plt.rcParams['xtick.labelsize'] = 'medium'
            plt.rcParams['ytick.labelsize'] = 'medium'

            # matplotlib.font_manager.FontProperties ---> matplotlib.rcParams
            plt.rcParams['legend.loc'] = 'best'
            plt.rcParams['legend.frameon'] = False
            plt.rcParams['legend.fontsize'] = 'small'
            # plt.rcParams['legend.fontweight'] = 'bold'  #key error
            plt.rcParams['legend.handlelength'] = 1.5

            plt.rcParams['figure.figsize'] = figsize or (6, 6)
            plt.rcParams['figure.autolayout'] = False
            plt.rcParams['figure.subplot.right'] = 0.8
            # plt.rcParams['figure.constrained_layout.use'] = True
            plt.rcParams['figure.dpi'] = 150

            plt.rcParams['savefig.transparent'] = True

            result = _func(*args, **kwargs)

            return result

        return wrapper

    if func is None:
        return decorator

    else:
        return decorator(func)


def timer(func=None):
    """ Decorator to measure the execution time of a function.

    This decorator calculates the elapsed time of the decorated function
    and prints the result in seconds.

    Parameters
    ----------
    - func (callable, optional): The function to be decorated.

    Returns
    -------
    - callable or None: If `func` is provided, returns the decorated function;
      otherwise, returns a decorator to be applied later.

    Examples
    --------
    1. Applying the decorator to a function directly:

    >>> @timer
    >>> def my_function():
    >>>     # Function logic here

    2. Using the decorator without providing a function:

    >>> @timer()
    >>> def my_function():
    >>>     # Function logic here

    Note
    ----
    The elapsed time is printed to the console after the function execution.

    """

    def decorator(_func):
        @wraps(_func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = _func(*args, **kwargs)
            end = time.perf_counter() - start
            print(f'{func.__name__} cost {end:.2f} seconds')
            return result

        return wrapper

    if func is None:
        return decorator

    else:
        return decorator(func)
