import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.pyplot import Axes
from matplotlib.ticker import AutoMinorLocator
from typing import Literal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from DataPlot.plot.core import *
from DataPlot.process import *


__all__ = [
        'contourmap',
]


@set_figure
def contourmap(DataBase) -> Axes:
    fig, ax = plt.subplots()

    npoints = 1000
    xreg = np.linspace(DataBase.PM25.min(), DataBase.PM25.max(), 83)
    yreg = np.linspace(DataBase.gRH.min(), DataBase.gRH.max(), 34)
    X, Y = np.meshgrid(xreg, yreg)

    d_f = DataBase.copy()
    DataBase['gRH'] = d_f['gRH'].round(2)
    DataBase['PM25'] = d_f['PM25'].round(2)

    def func(para, a, b):
        PM, GF = para
        return a * (PM * GF) ** (b)

    fit_df = DataBase[['PM25', 'gRH', 'Extinction']].dropna()
    popt, pcov = curve_fit(func, xdata=(fit_df['PM25'], fit_df['gRH']), ydata=fit_df['Extinction'])
    # print(popt)

    def f(x, y):
        return popt[0] * (x * y) ** (popt[1])

    plt.xlabel(Unit('PM25'))
    plt.ylabel('GF(RH)')
    plt.xlim(DataBase.PM25.min(), DataBase.PM25.max())
    plt.ylim(DataBase.gRH.min(), DataBase.gRH.max())
    plt.title('')

    # pcolor = ax.pcolormesh(X, Y, (X * 4.5 * Y ** (1 / 3)), cmap='jet', shading='auto', vmin=0, vmax=843, alpha=0.8)
    cont = ax.contour(X, Y, f(X, Y), colors='black', levels=5, vmin=0, vmax=f(X, Y).max(), linewidths=2)
    conf = ax.contourf(X, Y, f(X, Y), cmap='YlGnBu', levels=100, vmin=0, vmax=f(X, Y).max())
    ax.clabel(cont, colors=['black'], fmt=lambda s: f"{s:.0f} 1/Mm")

    cax = inset_axes(ax, width="5%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)

    color_bar = plt.colorbar(conf, cax=cax)
    color_bar.set_label(label='Extinction (1/Mm)')
    color_bar.ax.set_xticklabels(color_bar.ax.get_xticks().astype(int))

    return ax
