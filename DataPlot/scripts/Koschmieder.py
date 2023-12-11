import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from DataPlot.templates import set_figure

Path_Data = Path('/Data-example')
df = pd.read_excel(Path_Data / 'Koschmieder.xlsx', sheet_name=0)
# x = Visibility, y = Extinction, log-log fit!!


def log_fit(x, y, func = lambda x, a: -x + a):
    x_log = np.log(x)
    y_log = np.log(y)

    popt, pcov = curve_fit(func, x_log, y_log)

    residuals  = y_log - func(x_log, *popt)
    ss_res    = np.sum(residuals ** 2)
    ss_total  = np.sum((y_log - np.mean(y_log)) ** 2)
    r_squared = 1 - (ss_res / ss_total)

    print(f'Const_Log = {popt.round(3)}')
    print(f'Const = {math.exp(popt).__round__(3)}')
    print(f'R^2 = {r_squared.__round__(3)}')
    return popt, pcov


def reciprocal_fit(x, y, func = lambda x, a, b : a / (x**b)):
    popt, pcov = curve_fit(func, x, y)

    residuals  = y - func(x, *popt)
    ss_res    = np.sum(residuals ** 2)
    ss_total  = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_total)

    print(f'Const = {popt.round(3)}')
    print(f'  R^2 = {r_squared.__round__(3)}')
    return popt, pcov


@set_figure
def kos_naked(df):
    _df1 = df[['Extinction_dry', 'Ext_gas', 'Vis_Naked']].dropna().copy()
    _df2 = df[['IMPROVE_ext', 'Ext_gas', 'Vis_Naked']].dropna().copy()

    x_data1 = _df1['Vis_Naked']
    y_data1 = _df1['Extinction_dry'] + _df1['Ext_gas']

    x_data2 = _df2['Vis_Naked']
    y_data2 = _df2['IMPROVE_ext'] + _df2['Ext_gas']

    # figure
    fig, axes = plt.subplots(1, 1, figsize=(5, 6), dpi=150, constrained_layout=True)

    para_coeff = []
    boxcolors = ['#3f83bf', '#a5bf6b']

    for i, (df_, x_data, y_data) in enumerate(zip([_df1, _df2], [x_data1, x_data2], [y_data1, y_data2])):
        df_['Total_Ext'] = y_data
        df_grp = df_.groupby('Vis_Naked')

        vals, median_vals, vis = [], [], []
        for j, (name, subdf) in enumerate(df_grp):
            if len(subdf['Total_Ext'].dropna()) > 20:
                vis.append('{:.0f}'.format(name))
                vals.append(subdf['Total_Ext'].dropna().values)
                median_vals.append(subdf['Total_Ext'].dropna().median())

        plt.boxplot(vals, labels=vis, positions=np.array(vis, dtype='int'), widths=0.4,
                    showfliers=False, showmeans=True, meanline=False, patch_artist=True,
                    boxprops=dict(facecolor=boxcolors[i], alpha=.7),
                    meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='k', markersize=4),
                    medianprops=dict(color='#000000', ls='-'))

        plt.scatter(x_data, y_data, marker='.', s=10, facecolor='white', edgecolor=boxcolors[i], alpha=0.1)

        # fit curve
        x = np.array(vis, dtype='float')
        y = np.array(median_vals, dtype='float')

        # func = lambda x, a: a / x
        # coeff, pcov = log_fit(x, y)

        func = lambda x, a, b: a / (x ** b)
        coeff, pcov = reciprocal_fit(x, y)

        para_coeff.append(coeff)

    # Plot lines (ref & Measurement)
    x_fit = np.linspace(0.1, 70, 1000)
    # line1, = axes.core(x_fit, func(x_fit, math.exp(para_coeff[0])), c='b', lw=3)
    # line2, = axes.core(x_fit, func(x_fit, math.exp(para_coeff[1])), c='g', lw=3)
    line1, = axes.plot(x_fit, func(x_fit, *para_coeff[0]), c='b', lw=3)
    line2, = axes.plot(x_fit, func(x_fit, *para_coeff[1]), c='g', lw=3)

    plt.legend(handles=[line1, line2],
               # labels=[r'$\bf Vis\ (km)\ =\ ' + '{:.0f}'.format(math.exp(para_coeff[0])) + '\ /\ Ext\ (Dry\ Extinction)$',
               #         r'$\bf Vis\ (km)\ =\ ' + '{:.0f}'.format(math.exp(para_coeff[1])) + '\ /\ Ext\ (Amb\ Extinction)$'],
               labels=[r'$\bf Ext\ =\ ' + '{:.0f}\ /\ Vis^{{{:.3f}}}'.format(*para_coeff[0]) + '\ (Dry\ Extinction)$',
                       r'$\bf Ext\ =\ ' + '{:.0f}\ /\ Vis^{{{:.3f}}}'.format(*para_coeff[1]) + '\ (Amb\ Extinction)$'],
               handlelength=1.5, loc='upper right', prop=dict(size=12), frameon=False, bbox_to_anchor=(0.99, 0.99))

    plt.xticks(ticks=np.array(range(0, 51, 5)), labels=np.array(range(0, 51, 5)))
    plt.xlim(0, 20)
    plt.ylim(0, 700)
    plt.title(r'$\bf Koschmieder\ relationship$')
    plt.xlabel(r'$\bf Naked\ Visibility\ (km)$')
    plt.ylabel(r'$\bf Extinction\ coefficient\ (1/Mm)$')
    plt.show()
    # fig.savefig(pth(f'Koschmieder_Naked'))


@set_figure
def kos_LPV(df,):
    _df1 = df[['Extinction_dry', 'Ext_gas', 'Vis_LPV']].dropna().copy()
    _df2 = df[['IMPROVE_ext', 'Ext_gas', 'Vis_LPV']].dropna().copy()

    x_data1 = _df1['Vis_LPV']
    y_data1 = _df1['Extinction_dry'] + _df1['Ext_gas']

    x_data2 = _df2['Vis_LPV']
    y_data2 = _df2['IMPROVE_ext'] + _df2['Ext_gas']

    # figure
    fig, axes = plt.subplots(1, 1, figsize=(5, 6), dpi=150, constrained_layout=True)

    para_coeff = []
    boxcolors = ['#3f83bf', '#a5bf6b']

    for i, (df_, x_data, y_data) in enumerate(zip([_df1, _df2], [x_data1, x_data2], [y_data1, y_data2])):
        df_['Total_Ext'] = y_data

        bins = np.linspace(0, 70, 36)
        wid = (bins + (bins[1] - bins[0]) / 2)[0:-1]

        df_[f'{x_data.name}' + '_bins'] = pd.cut(x=x_data, bins=bins, labels=wid)

        grouped = df_.groupby(f'{x_data.name}' + '_bins')

        vals, median_vals, vis = [], [], []
        for j, (name, subdf) in enumerate(grouped):
            if len(subdf['Total_Ext'].dropna()) > 20:
                vis.append('{:.1f}'.format(name))
                vals.append(subdf['Total_Ext'].dropna().values)
                median_vals.append(subdf['Total_Ext'].dropna().mean())

        plt.boxplot(vals, labels=vis, positions=np.array(vis, dtype='float'), widths=(bins[1] - bins[0])/2.5,
                    showfliers=False, showmeans=True, meanline=False, patch_artist=True,
                    boxprops=dict(facecolor=boxcolors[i], alpha=.7),
                    meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='k', markersize=4),
                    medianprops=dict(color='#000000', ls='-'))

        plt.scatter(x_data, y_data, marker='.', s=10, facecolor='white', edgecolor=boxcolors[i], alpha=0.1)

        # fit curve
        x = np.array(vis, dtype='float')
        y = np.array(median_vals, dtype='float')

        # coeff, pcov = log_fit(x, y)
        # para_coeff.append(coeff)

        func = lambda x, a, b: a / (x**b)
        coeff, pcov = reciprocal_fit(x, y)
        para_coeff.append(coeff)
    # Plot lines (ref & Measurement)

    # func = lambda x, a: a / x
    x_fit = np.linspace(0.1, 70, 1000)
    # line1, = axes.core(x_fit, func(x_fit, math.exp(para_coeff[0])), c='b', lw=3)
    # line2, = axes.core(x_fit, func(x_fit, math.exp(para_coeff[1])), c='g', lw=3)
    line1, = axes.plot(x_fit, func(x_fit, *para_coeff[0]), c='b', lw=3)
    line2, = axes.plot(x_fit, func(x_fit, *para_coeff[1]), c='g', lw=3)

    plt.legend(handles=[line1, line2],
               # labels=[r'$\bf Vis\ (km)\ =\ ' + '{:.0f}'.format(math.exp(para_coeff[0])) + '\ /\ Ext\ (Dry\ Extinction)$',
               #         r'$\bf Vis\ (km)\ =\ ' + '{:.0f}'.format(math.exp(para_coeff[1])) + '\ /\ Ext\ (Amb\ Extinction)$'],
               labels=[r'$\bf Ext\ =\ ' + '{:.0f}\ /\ Vis^{{{:.3f}}}'.format(*para_coeff[0]) + '\ (Dry\ Extinction)$',
                       r'$\bf Ext\ =\ ' + '{:.0f}\ /\ Vis^{{{:.3f}}}'.format(*para_coeff[1]) + '\ (Amb\ Extinction)$'],
               handlelength=1.5, loc='upper right', prop=dict(size=12), frameon=False, bbox_to_anchor=(0.99, 0.99))
    plt.xticks(ticks=np.array(range(0, 71, 5)), labels=np.array(range(0, 71, 5)))
    plt.xlim(0, 50)
    plt.ylim(0, 700)
    plt.title(r'$\bf Koschmieder\ relationship$')
    plt.xlabel(r'$\bf LPV\ Visibility\ (km)$')
    plt.ylabel(r'$\bf Extinction\ coefficient\ (1/Mm)$')
    plt.show()
    # fig.savefig(pth(f'Koschmieder_LPV'))


if __name__ == '__main__':
    kos_naked(df)
    # kos_LPV(df)
