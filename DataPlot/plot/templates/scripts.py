import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.ticker import AutoMinorLocator
from typing import Literal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from DataPlot.plot.core import *
from DataPlot.process import *


@set_figure(figsize=(6, 6), fs=8)
def corr_matrix(data: pd.DataFrame,
                cmap: str = "RdBu",
                ax: plt.Axes | None = None,
                ) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()

    columns = ['Extinction', 'Scattering', 'Absorption', 'PM1', 'PM25', 'PM10', 'PBLH', 'VC',
               'AT', 'RH', 'WS', 'NO', 'NO2', 'NOx', 'O3', 'Benzene', 'Toluene',
               'SO2', 'CO', 'THC', 'CH4', 'NMHC', 'NH3', 'HCl', 'HNO2', 'HNO3',
               'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+', 'Cl-', 'NO2-', 'NO3-', 'SO42-', ]

    df = data[columns]

    _corr = df.corr()
    corr = pd.melt(_corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']

    p_values = _corr.apply(lambda col1: _corr.apply(lambda col2: pearsonr(col1, col2)[1]))
    p_values = p_values.mask(p_values > 0.05)
    p_values = pd.melt(p_values.reset_index(), id_vars='index').dropna()
    p_values.columns = ['x', 'y', 'value']

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(corr['x'].unique())]
    y_labels = [v for v in sorted(corr['y'].unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=90, horizontalalignment='center')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    # ax.tick_params(axis='both', which='major', direction='out', top=True, left=True)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    n_colors = 256  # Use 256 colors for the diverging color palette
    palette = sns.color_palette(cmap, n_colors=n_colors)  # Create the palette

    # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    point = ax.scatter(
        x=corr['x'].map(x_to_num),
        y=corr['y'].map(y_to_num),
        s=corr['value'].abs() * 70,
        c=corr['value'].apply(value_to_color),  # Vector of square color values, mapped to color palette
        marker='s',
        label='$R^{2}$'
    )

    axes_image = plt.cm.ScalarMappable(cmap=colormaps[cmap])

    cax = inset_axes(ax, width="5%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    plt.subplots_adjust(bottom=0.15, right=0.8)
    cbar = plt.colorbar(mappable=axes_image, cax=cax, label=r'$R^{2}$')

    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(np.linspace(-1, 1, 5))

    point2 = ax.scatter(
        x=p_values['x'].map(x_to_num),
        y=p_values['y'].map(y_to_num),
        s=10,
        marker='*',
        color='k',
        label='p < 0.05'
    )

    ax.legend(handles=[point2], labels=['p < 0.05'], bbox_to_anchor=(0.05, 1, 0.1, 0.05))
    plt.show()

    return ax


@set_figure(figsize=(4, 4), fs=8)
def diurnal_pattern(data_set: pd.DataFrame,
                    data_std: pd.DataFrame,
                    y: str | list[str],
                    std_area=0.5,
                    ax: plt.Axes | None = None,
                    **kwargs) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()

    linecolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']

    Hour = range(0, 24)

    mean = data_set[y]
    std = data_std[y] * std_area

    # Plot Diurnal pattern
    ax.plot(Hour, mean, 'r')
    ax.fill_between(Hour, y1=mean + std, y2=mean - std, alpha=0.5, color=linecolors[0], linewidth=2, edgecolor=None)

    ax.set(xlabel=kwargs.get('xlabel', 'Hours'),
           ylabel=kwargs.get('ylabel', Unit(y)),
           xlim=kwargs.get('xlim', (0, 23)),
           ylim=kwargs.get('ylim', (None, None)),
           xticks=kwargs.get('xticks', [0, 4, 8, 12, 16, 20]))

    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='x', which='minor')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=True)
    plt.show()

    return ax


@set_figure(fs=12)
def koschmieder(df: pd.DataFrame,
                y: Literal['Vis_Naked', 'Vis_LPV'],
                function: Literal['log', 'reciprocal'] = 'log',
                ax: plt.Axes | None = None,
                **kwargs):  # x = Visibility, y = Extinction, log-log fit!!
    def _log_fit(x, y, func=lambda x, a: -x + a):
        x_log = np.log(x)
        y_log = np.log(y)

        popt, pcov = curve_fit(func, x_log, y_log)

        residuals = y_log - func(x_log, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_total = np.sum((y_log - np.mean(y_log)) ** 2)
        r_squared = 1 - (ss_res / ss_total)
        print(f'Const_Log = {popt[0].round(3)}')
        print(f'Const = {np.exp(popt)[0].round(3)}')
        print(f'R^2 = {r_squared.round(3)}')
        return np.exp(popt)[0], pcov

    def _reciprocal_fit(x, y, func=lambda x, a, b: a / (x ** b)):
        popt, pcov = curve_fit(func, x, y)

        residuals = y - func(x, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_total)
        print(f'Const = {popt.round(3)}')
        print(f'  R^2 = {r_squared.round(3)}')
        return popt, pcov

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    _df1 = df[['Extinction', 'ExtinctionByGas', y]].dropna().copy()
    _df2 = df[['total_ext_dry', 'ExtinctionByGas', y]].dropna().copy()

    x_data1 = _df1[y]
    y_data1 = _df1['Extinction'] + _df1['ExtinctionByGas']

    x_data2 = _df2[y]
    y_data2 = _df2['total_ext_dry'] + _df2['ExtinctionByGas']

    para_coeff = []
    boxcolors = ['#3f83bf', '#a5bf6b']

    for i, (df_, x_data, y_data) in enumerate(zip([_df1, _df2], [x_data1, x_data2], [y_data1, y_data2])):
        df_['Total_Ext'] = y_data

        if y == 'Vis_Naked':
            df_grp = df_.groupby(f'{y}')

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

        if y == 'Vis_LPV':
            bins = np.linspace(0, 70, 36)
            wid = (bins + (bins[1] - bins[0]) / 2)[0:-1]

            df_[f'{x_data.name}' + '_bins'] = pd.cut(x=x_data, bins=bins, labels=wid)

            grouped = df_.groupby(f'{x_data.name}' + '_bins', observed=False)

            vals, median_vals, vis = [], [], []
            for j, (name, subdf) in enumerate(grouped):
                if len(subdf['Total_Ext'].dropna()) > 20:
                    vis.append('{:.1f}'.format(name))
                    vals.append(subdf['Total_Ext'].dropna().values)
                    median_vals.append(subdf['Total_Ext'].dropna().mean())

            plt.boxplot(vals, labels=vis, positions=np.array(vis, dtype='float'), widths=(bins[1] - bins[0]) / 2.5,
                        showfliers=False, showmeans=True, meanline=False, patch_artist=True,
                        boxprops=dict(facecolor=boxcolors[i], alpha=.7),
                        meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='k', markersize=4),
                        medianprops=dict(color='#000000', ls='-'))

            plt.scatter(x_data, y_data, marker='.', s=10, facecolor='white', edgecolor=boxcolors[i], alpha=0.1)

        # fit curve
        _x = np.array(vis, dtype='float')
        _y = np.array(median_vals, dtype='float')

        if function == 'log':
            func = lambda x, a: a / x
            coeff, pcov = _log_fit(_x, _y)

        else:
            func = lambda x, a, b: a / (x ** b)
            coeff, pcov = _reciprocal_fit(_x, _y)

        para_coeff.append(coeff)

    # Plot lines (ref & Measurement)
    x_fit = np.linspace(0.1, 70, 1000)

    if function == 'log':
        line1, = ax.plot(x_fit, func(x_fit, para_coeff[0]), c='b', lw=3)
        line2, = ax.plot(x_fit, func(x_fit, para_coeff[1]), c='g', lw=3)

        labels = ['Vis (km) = ' + f'{round(para_coeff[0])}' + ' / Ext (Dry Extinction)',
                  'Vis (km) = ' + f'{round(para_coeff[1])}' + ' / Ext (Amb Extinction)']

    else:
        x_fit = np.linspace(0.1, 70, 1000)
        line1, = ax.plot(x_fit, func(x_fit, *para_coeff[0]), c='b', lw=3)
        line2, = ax.plot(x_fit, func(x_fit, *para_coeff[1]), c='g', lw=3)

        labels = [f'Ext = ' + '{:.0f} / Vis ^ {:.3f}'.format(*para_coeff[0]) + ' (Dry Extinction)',
                  f'Ext = ' + '{:.0f} / Vis ^ {:.3f}'.format(*para_coeff[1]) + ' (Amb Extinction)']

    plt.legend(handles=[line1, line2], labels=labels, loc='upper right', prop=dict(size=10, weight='bold'),
               bbox_to_anchor=(0.99, 0.99))

    plt.xticks(ticks=np.array(range(0, 51, 5)), labels=np.array(range(0, 51, 5)))
    plt.xlim(0, 50)
    plt.ylim(0, 700)
    plt.title(r'$\bf Koschmieder\ relationship$')
    plt.xlabel(f'{y} (km)')
    plt.ylabel(r'$\bf Extinction\ coefficient\ (1/Mm)$')
    plt.show()

    return ax


@set_figure(figsize=(6, 5))
def gf_pm_ext():
    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.8)
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
    color_bar.set_label(label='Extinction (1/Mm)', weight='bold', size=16)
    color_bar.ax.set_xticklabels(color_bar.ax.get_xticks().astype(int), size=16)
    plt.show()


@set_figure(figsize=(6, 5))
def four_quar():
    subdf = DataBase[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna().resample('3h').mean()

    item = 'RH'
    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.8)
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

    ax.legend(loc='upper right', bbox_to_anchor=(0.8, 0.8, 0.2, 0.2), title=Unit('RH'))

    plt.show()


if __name__ == '__main__':
    # koschmieder(DataBase, 'Vis_LPV', 'log')
    # koschmieder(DataBase, 'Vis_Naked', 'reciprocal')
    four_quar()
    gf_pm_ext()
