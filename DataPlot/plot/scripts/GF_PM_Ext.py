import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot import set_figure, Unit
from DataPlot.process import DataBase


subdf = DataBase[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna().resample('3h').mean()


@set_figure(figsize=(7, 6))
def gf_pm_ext():
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

    def fmt(x):
        s = f"{x:.0f} 1/Mm"
        return rf"{s}"

    plt.xlabel(Unit('PM25'))
    plt.ylabel('GF(RH)')
    plt.xlim(DataBase.PM25.min(), DataBase.PM25.max())
    plt.ylim(DataBase.gRH.min(), DataBase.gRH.max())
    plt.title('')

    # pcolor = ax.pcolormesh(X, Y, (X * 4.5 * Y ** (1 / 3)), cmap='jet', shading='auto', vmin=0, vmax=843, alpha=0.8)
    cont = ax.contour(X, Y, f(X, Y), colors='black', levels=5, vmin=0, vmax=f(X, Y).max(), linewidths=2)
    conf = ax.contourf(X, Y, f(X, Y), cmap='YlGnBu', levels=100, vmin=0, vmax=f(X, Y).max())
    ax.clabel(cont, colors=['black'], fmt=fmt, fontsize=16)

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


@set_figure(figsize=(8, 6))
def four_quar(subdf):
    item = 'RH'
    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(subdf['PM25'], subdf['Vis_LPV'], s=50 * (subdf[item] / subdf[item].max()) ** 4, c=subdf['VC'],
                    norm=plt.Normalize(vmin=0, vmax=2000), cmap='YlGnBu')

    axins = inset_axes(ax, width="5%", height="100%", loc='lower left',
                       bbox_to_anchor=(1.02, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    color_bar = plt.colorbar(sc, cax=axins, orientation='vertical')
    color_bar.set_label(label=Unit('VC'))

    ax.tick_params(axis='x', which='major', direction="out", length=6)
    ax.tick_params(axis='y', which='major', direction="out", length=6)
    ax.set_xlim(0., 80)
    ax.set_ylim(0., 50)
    ax.set_ylabel(r'$\bf Visibility\ (km)$')
    ax.set_xlabel(r'$\bf PM_{2.5}\ (\mu g/m^3)$')

    dot = np.linspace(subdf[item].min(), subdf[item].max(), 6).round(-1)

    for dott in dot[1:-1]:
        ax.scatter([], [], c='k', alpha=0.8, s=200 * (dott / subdf[item].max()) ** 4, label='{:.0f}'.format(dott))

    ax.legend(loc='upper right', bbox_to_anchor=(0.8, 0.8, 0.2, 0.2), scatterpoints=1, frameon=False, labelspacing=0.5,
              title=Unit('RH'))

    plt.show()


if __name__ == '__main__':
    four_quar(subdf)
    gf_pm_ext()


