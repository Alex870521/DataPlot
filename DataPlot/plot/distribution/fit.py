import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from numpy import log, exp, pi, sqrt
from tabulate import tabulate
from DataPlot.plot.core import *


def curvefit(dp, dist, mode=None, **kwargs):
    """
    Fit a log-normal distribution to the given data and plot the result.

    Parameters
    ----------
    - dp (array): Array of diameter values.
    - dist (array): Array of distribution values corresponding to each diameter.
    - mode (int, optional): Number of log-normal distributions to fit (default is None).
    - **kwargs: Additional keyword arguments to be passed to the plot_function.

    Returns
    -------
    None

    Notes
    -----
    - The function fits a sum of log-normal distributions to the input data.
    - The number of distributions is determined by the 'mode' parameter.
    - Additional plotting customization can be done using the **kwargs.

    Example
    -------
    >>> curvefit(dp, dist, mode=2, xlabel="Diameter (nm)", ylabel="Distribution", figname="extinction")
    """
    # Calculate total number concentration and normalize distribution
    total_num = np.sum(dist * log(dp))
    norm_data = dist / total_num

    def lognorm_func(x, *params):
        num_distributions = len(params) // 3
        result = np.zeros_like(x)

        for i in range(num_distributions):
            offset = i * 3
            _number = params[offset]
            _geomean = params[offset + 1]
            _geostd = params[offset + 2]
            result += (_number / (log(_geostd) * sqrt(2 * pi)) *
                       exp(-(log(x) - log(_geomean)) ** 2 / (2 * log(_geostd) ** 2)))

        return result

    # initial gauss
    min_value = np.array([min(dist)])
    extend_ser = np.concatenate([min_value, dist, min_value])
    _mode, _ = find_peaks(extend_ser, distance=20)
    peak = dp[_mode - 1]
    mode = mode or len(peak)

    # 設定參數範圍
    bounds = ([1e-6, 10, 1] * mode,
              [1, 2500, 8] * mode)

    # 初始參數猜測
    initial_guess = [0.05, 20, 2] * mode

    # 使用 curve_fit 函數進行擬合
    popt, pcov = curve_fit(lognorm_func, dp, norm_data,
                           p0=initial_guess,
                           maxfev=2000000,
                           method='trf',
                           bounds=bounds)

    # 獲取擬合的參數
    params = popt.tolist()

    print("擬合結果:")
    table = []

    for i in range(mode):
        offset = i * 3
        num, mu, sigma = params[offset:offset + 3]
        table.append([f'log-{i + 1}', num * num, mu, sigma])

    formatted_data = [[item if not isinstance(item, float) else f"{item:.3f}" for item in row] for row in table]

    # 使用 tabulate 來建立表格並印出
    tab = tabulate(formatted_data, headers=["log-", "number", "mu", "sigma"], floatfmt=".3f", tablefmt="fancy_grid")
    print(tab)

    @set_figure
    def plot_function(dp, observed, fit_curve, **kwargs):
        fig, ax = plt.subplots()

        plt.plot(dp, fit_curve, color='#c41b1b', label='Fitting curve', lw=2.5)
        plt.plot(dp, observed, color='b', label='Observed curve', lw=2.5)

        xlim = kwargs.get('xlim') or (11.8, 2500)
        ylim = kwargs.get('ylim') or (0, None)
        xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
        ylabel = kwargs.get('ylabel') or r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useMathText=True)
        ax.set_title('')
        ax.legend(loc='best', frameon=False)
        figname = kwargs.get('figname') or ''
        plt.semilogx()
        # plt.savefig(f'CurveFit_{figname}.png')
        plt.show()

    # plot result
    plot_function(dp, dist, total_num * lognorm_func(dp, *params), **kwargs)
