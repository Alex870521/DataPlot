import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes
from scipy.optimize import curve_fit

from DataPlot.plot.core import *

__all__ = [
    'contour',
]


@set_figure
def contour(DataBase) -> Axes:
    fig, ax = plt.subplots()

    npoints = 1000
    xreg = np.linspace(DataBase.PM25.min(), DataBase.PM25.max(), 83)
    yreg = np.linspace(DataBase.gRH.min(), DataBase.gRH.max(), 34)
    X, Y = np.meshgrid(xreg, yreg)

    d_f = DataBase.copy()
    DataBase['gRH'] = d_f['gRH'].round(2)
    DataBase['PM25'] = d_f['PM25'].round(2)

    def func(data, *params):
        return params[0] * data ** (params[1])

    initial_guess = [1.0, 1.0]

    fit_df = DataBase[['PM25', 'gRH', 'Extinction']].dropna()
    popt, pcov = curve_fit(func, xdata=(fit_df['PM25'] * fit_df['gRH']), ydata=fit_df['Extinction'], p0=initial_guess,
                           maxfev=2000000, method='trf')

    x, y = DataBase.PM25, DataBase.gRH

    # pcolor = ax.pcolormesh(X, Y, (X * 4.5 * Y ** (1 / 3)), cmap='jet', shading='auto', vmin=0, vmax=843, alpha=0.8)
    Z = func(X * Y, *popt)
    cont = ax.contour(X, Y, Z, colors='black', levels=5, vmin=0, vmax=Z.max())
    conf = ax.contourf(X, Y, Z, cmap='YlGnBu', levels=100, vmin=0, vmax=Z.max())
    ax.clabel(cont, colors=['black'], fmt=lambda s: f"{s:.0f} 1/Mm")
    ax.set(xlabel=Unit('PM25'), ylabel=Unit('gRH'), xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))

    color_bar = plt.colorbar(conf, pad=0.02, fraction=0.05, label='Extinction (1/Mm)')
    color_bar.ax.set_xticklabels(color_bar.ax.get_xticks().astype(int))

    return ax


if __name__ == '__main__':
    from DataPlot import *

    contour(DataBase)
