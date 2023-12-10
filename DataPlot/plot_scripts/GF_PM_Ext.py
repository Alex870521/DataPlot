from pathlib import Path
from pandas import read_csv, concat, date_range
from DataPlot.data_processing import main
from DataPlot.data_processing.Data_classify import state_classify, season_classify, Seasons
from datetime import datetime
import pandas as pd
import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot_templates import set_figure, unit, getColor, color_maker


df = main()


@set_figure
def gf_pm_ext():
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=0)

    ax.set_position([0.125,0.125,0.65,0.8])

    npoints=1000
    xreg = np.linspace(df.PM25.min(),df.PM25.max(),83)
    yreg = np.linspace(df.gRH.min(),df.gRH.max(),34)
    X,Y = np.meshgrid(xreg,yreg)

    d_f = df.copy()
    df['gRH'] = d_f['gRH'].round(2)
    df['PM25'] = d_f['PM25'].round(2)
    table = df.pivot_table(index=['PM25'], columns=['gRH'], values=['Extinction'], aggfunc=np.mean)

    def func(para, a, b):
        PM, GF = para
        return a * (PM * GF)**(b)


    fit_df = df[['PM25', 'gRH', 'Extinction']].dropna()
    popt, pcov = curve_fit(func, (fit_df['PM25'], fit_df['gRH']), fit_df['Extinction'])
    print(popt)
    def f(x, y):
        return popt[0]*(x*y)**(popt[1])

    def fmt(x):
        s = f"{x:.0f} 1/Mm"
        return rf"{s}"

    plt.xlabel(unit('PM25'))
    plt.ylabel('GF(RH)')
    plt.xlim(df.PM25.min(),df.PM25.max())
    plt.ylim(df.gRH.min(),df.gRH.max())
    plt.title('')

    # pcolor =  ax.pcolormesh(X, Y, (X*4.5*Y**(1/3)), cmap= 'jet',shading='auto',vmin=0,vmax = 843, alpha=0.8)
    cont = ax.contour(X, Y, f(X,Y), colors='black', levels=5, vmin=0, vmax=f(X,Y).max(), linewidths=2)
    conf = ax.contourf(X, Y, f(X,Y), cmap='YlGnBu', levels=100, vmin=0, vmax=f(X,Y).max())
    ax.clabel(cont, colors=['black'], fmt=fmt, fontsize=16)

    plt.scatter(df['PM25'], df['gRH'], c=df.Extinction, norm=plt.Normalize(vmin=df.Extinction.min(), vmax=df.Extinction.max()), cmap='jet',
                  marker='o', s=20, facecolor="b", edgecolor=None, alpha=0.5)

    box = ax.get_position()
    cax = fig.add_axes([0.8, box.y0, 0.03, box.y1-box.y0])
    color_bar = plt.colorbar(conf, cax=cax)
    color_bar.set_label(label='Extinction (1/Mm)',family='Times New Roman', weight='bold',size=16)
    color_bar.ax.set_xticklabels(color_bar.ax.get_xticks().astype(int), size=16)
    plt.show()

gf_pm_ext()

plt.scatter()