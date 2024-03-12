import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot import set_figure, Unit
from DataPlot.process import DataBase


df = DataBase


@set_figure(figsize=(7, 6))
def gf_pm_ext():
    fig, ax = plt.subplots()

    npoints = 1000
    xreg = np.linspace(df.PM25.min(), df.PM25.max(), 83)
    yreg = np.linspace(df.gRH.min(), df.gRH.max(), 34)
    X, Y = np.meshgrid(xreg, yreg)

    d_f = df.copy()
    df['gRH'] = d_f['gRH'].round(2)
    df['PM25'] = d_f['PM25'].round(2)

    def func(para, a, b):
        PM, GF = para
        return a * (PM * GF) ** (b)

    fit_df = df[['PM25', 'gRH', 'Extinction']].dropna()
    popt, pcov = curve_fit(func, (fit_df['PM25'], fit_df['gRH']), fit_df['Extinction'])
    print(popt)

    def f(x, y):
        return popt[0] * (x * y) ** (popt[1])

    def fmt(x):
        s = f"{x:.0f} 1/Mm"
        return rf"{s}"

    plt.xlabel(Unit('PM25'))
    plt.ylabel('GF(RH)')
    plt.xlim(df.PM25.min(), df.PM25.max())
    plt.ylim(df.gRH.min(), df.gRH.max())
    plt.title('')

    # pcolor = ax.pcolormesh(X, Y, (X * 4.5 * Y ** (1 / 3)), cmap='jet', shading='auto', vmin=0, vmax=843, alpha=0.8)
    cont = ax.contour(X, Y, f(X, Y), colors='black', levels=5, vmin=0, vmax=f(X, Y).max(), linewidths=2)
    conf = ax.contourf(X, Y, f(X, Y), cmap='YlGnBu', levels=100, vmin=0, vmax=f(X, Y).max())
    ax.clabel(cont, colors=['black'], fmt=fmt, fontsize=16)

    plt.scatter(df['PM25'], df['gRH'], c=df.Extinction,
                norm=plt.Normalize(vmin=df.Extinction.min(), vmax=df.Extinction.max()), cmap='jet',
                marker='o', s=20, facecolor="b", edgecolor=None, alpha=0.5)

    cax = inset_axes(ax, width="3%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)

    color_bar = plt.colorbar(conf, cax=cax)
    color_bar.set_label(label='Extinction (1/Mm)', family='Times New Roman', weight='bold', size=16)
    color_bar.ax.set_xticklabels(color_bar.ax.get_xticks().astype(int), size=16)
    plt.show()


gf_pm_ext()
