import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_templates import set_figure, unit, getColor
from sklearn.linear_model import LinearRegression
import seaborn as sns

# ref https://seaborn.pydata.org/generated/seaborn.scatterplot.html

__all__ = ['scatter',
           'scatter_mutiReg']


def _range(series, **kwargs):
    _data = np.array(series)
    max_value, min_value = _data.max() * 1.1, _data.min() * 0.9
    data_range = (min_value, max_value)
    range_from_kwargs = kwargs.get('range')
    return _data, max_value, min_value, range_from_kwargs or data_range


@set_figure(figsize=(6, 5))
def scatter(_df, x, y, c=None, s=None, cmap='jet', regression=None, diagonal=False, box=False, **kwargs):

    df = _df.dropna(subset=[x, y])
    x_data, x_max, x_min, x_range = _range(df[x], range=kwargs.get('x_range'))
    y_data, y_max, y_min, y_range = _range(df[y], range=kwargs.get('y_range'))

    fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize'))  # None = reParams

    if c is not None and s is not None:
        c_data, c_max, c_min, c_range = _range(df[c], range=kwargs.get('c_range'))
        s_data, s_max, s_min, s_range = _range(df[s], range=kwargs.get('s_range'))

        scatter = ax.scatter(x_data, y_data, c=c_data, vmin=c_range[0], vmax=c_range[1], cmap=cmap, s=s_data, alpha=0.7, edgecolors=None)
        colorbar = True

        dot = np.linspace(s_range[0], s_range[1], 6).round(-1)

        for dott in dot[1:-1]:
            plt.scatter([], [], c='k', alpha=0.8, s=300 * (dott / s_data.max()) ** 1.5, label='{:.0f}'.format(dott))

        plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5, title=unit(s))

    elif c is not None:
        c_data, c_max, c_min, c_range = _range(df[c], range=kwargs.get('c_range'))

        scatter = ax.scatter(x_data, y_data, c=c_data, vmin=c_range[0], vmax=c_range[1], cmap=cmap, alpha=0.7, edgecolors=None)
        colorbar = True

    elif s is not None:
        s_data, s_max, s_min, s_range = _range(df[s], range=kwargs.get('s_range'))

        scatter = ax.scatter(x_data, y_data, s=s_data, color='#7a97c9', alpha=0.7, edgecolors='white')
        colorbar = False

        # dealing
        dot = np.linspace(s_range[0], s_range[1], 6).round(-1)

        for dott in dot[1:-1]:
            plt.scatter([], [], c='k', alpha=0.8, s=300 * (dott / s_data.max()) ** 1.5, label='{:.0f}'.format(dott))

        plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5, title=unit(s))

    else:
        scatter = ax.scatter(x_data, y_data, s=30, color='#7a97c9', alpha=0.7, edgecolors='white')
        colorbar = False

    xlim = kwargs.get('xlim') or x_range
    ylim = kwargs.get('ylim') or y_range
    xlabel = kwargs.get('xlabel') or unit(x) or 'xlabel'
    ylabel = kwargs.get('ylabel') or unit(y) or 'ylabel'
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    title = kwargs.get('title') or 'title'
    ax.set_title(title, fontdict={'fontweight': 'bold', 'fontsize': 20})

    # color_bar
    if colorbar:
        color_bar = plt.colorbar(scatter, extend='both')
        color_bar.set_label(label=unit(c) or 'clabel', size=14)

    if regression:
        x_fit = x_data.reshape(-1, 1)
        y_fit = y_data.reshape(-1, 1)

        model = LinearRegression(fit_intercept=True).fit(x_fit, y_fit)

        slope = round(model.coef_[0][0], 3)
        intercept = round(model.intercept_[0], 3)
        r_square = round(model.score(x_fit, y_fit), 3)

        plt.plot(x_fit, model.predict(x_fit), linewidth=3, color=sns.xkcd_rgb["denim blue"], alpha=1)

        text = np.poly1d([slope, intercept])
        func = 'y = ' + str(text).replace('\n', "") + '\n' + r'$\bf R^2 = $' + str(r_square)
        plt.text(0.05, 0.95, f'{func}', fontdict={'weight': 'bold'}, color=sns.xkcd_rgb["denim blue"],
                 ha='left', va='top', transform=ax.transAxes)

    if diagonal:
        ax.axline((0, 0), slope=1., color='k', lw=2, ls='--', alpha=0.5, label='1:1')
        plt.text(0.91, 0.97, r'$\bf 1:1\ Line$', color='k', ha='right', va='top', transform=ax.transAxes)

    if box:
        bins = np.linspace(x_range[0], x_range[1], 11, endpoint=True)
        wid = (bins + (bins[1] - bins[0]) / 2)[0:-1]

        df[f'{x}' + '_bin'] = pd.cut(x=x_data, bins=bins, labels=wid)

        group = f'{x}' + '_bin'
        column = f'{y}'
        grouped = df.groupby(group)

        names, vals = [], []

        for i, (name, subdf) in enumerate(grouped):
            names.append('{:.0f}'.format(name))
            vals.append(subdf[column].dropna().values)

        plt.boxplot(vals, labels=names, positions=wid, widths=(bins[1] - bins[0]) / 3,
                    showfliers=False, showmeans=True, meanline=True, patch_artist=True,
                    boxprops=dict(facecolor='#f2c872', alpha=.7),
                    meanprops=dict(color='#000000', ls='none'),
                    medianprops=dict(ls='-', color='#000000'))

        plt.xlim(x_range[0], x_range[1])
        ax.set_xticks(bins, labels=bins.astype(int))

    # savefig

    return fig, ax


@set_figure(figsize=(6, 5))
def scatter_mutiReg(df, x, y1, y2, regression=None, diagonal=False, **kwargs):
    # create fig
    fig, ax = plt.subplots()

    df = df.dropna(subset=[x, y1, y2])

    x_data = np.array(df[x])

    y_data1 = np.array(df[y1])
    y_data2 = np.array(df[y2])

    from plot_templates import linecolor
    color1, color2, color3 = linecolor()

    scatter1 = ax.scatter(x_data, y_data1, s=25, color=color1['face'], alpha=0.8, edgecolors=color1['edge'], label='Internal')
    scatter2 = ax.scatter(x_data, y_data2, s=25, color=color2['face'], alpha=0.8, edgecolors=color2['edge'], label='External')

    xlim = kwargs.get('xlim')
    ylim = kwargs.get('ylim')
    xlabel = kwargs.get('xlabel') or unit(x) or ''
    ylabel = kwargs.get('ylabel') or unit(y1) or ''
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    title = kwargs.get('title') or ''
    ax.set_title(title, fontdict={'fontweight': 'bold', 'fontsize': 20})

    if regression:
        x_fit = x_data.reshape(-1, 1)
        y_fit = y_data1.reshape(-1, 1)

        model = LinearRegression().fit(x_fit, y_fit)

        slope = round(model.coef_[0][0], 3)
        intercept = round(model.intercept_[0], 3)
        r_square = round(model.score(x_fit, y_fit), 3)

        plt.plot(x_fit, model.predict(x_fit), linewidth=3, color=color1['line'], alpha=1, zorder=3)

        text = np.poly1d([slope, intercept])
        func = 'y = ' + str(text).replace('\n', "") + '\n' + r'$\bf R^2 = $' + str(r_square)
        plt.text(0.05, 0.75, f'{func}', fontdict={'weight': 'bold'}, color=color1['line'], ha='left', va='top',
                 transform=ax.transAxes)

        x_fit = x_data.reshape(-1, 1)
        y_2d2 = y_data2.reshape(-1, 1)
        model2 = LinearRegression().fit(x_fit, y_2d2)
        slope = model2.coef_[0][0].__round__(3)
        intercept = model2.intercept_[0].__round__(3)
        r_square = model2.score(x_fit, y_2d2).__round__(3)

        plt.plot(x_fit, model2.predict(x_fit), linewidth=3, color=color2['line'], alpha=1, zorder=3)

        text = np.poly1d([slope, intercept])
        func = 'y = ' + str(text).replace('\n', "") + '\n' + r'$\bf R^2 = $' + str(r_square)
        plt.text(0.05, 0.95, f'{func}', fontdict={'weight': 'bold'}, color=color2['line'], ha='left',
                 va='top',
                 transform=ax.transAxes)

    if diagonal:
        ax.axline((0, 0), slope=1., color='k', lw=2, ls='--', alpha=0.5, label='1:1')
        plt.text(0.97, 0.5, r'$\bf 1:1\ Line$', color='k', ha='right', va='top', transform=ax.transAxes)

    # plt.legend(handles=[scatter1, scatter2], loc='best', prop={'weight': 'bold'})

    # savefig

    return fig, ax


if __name__ == '__main__':
    df = pd.DataFrame(
        {'Extinction': np.linspace(0, 5, 500) + np.random.randn(500)*1*np.random.randn(500),
         'Scattering': np.linspace(0, 10, 500) + np.random.randn(500)*1*np.random.randn(500),
         'Absorption': np.linspace(0, 8, 500) + np.random.randn(500)*1*np.random.randn(500),
         'RH': np.random.random(500) * 100,
         'diversity': np.random.random(500) * 1})

    # scatter(df, x='Extinction', y='Scattering', c='RH', title='Title', regression=True, box=True)
    scatter_mutiReg(df, x='Extinction', y1='Scattering', y2='Absorption', regression=True)
    # sns.jointplot(df, x='Extinction', y='Scattering')
