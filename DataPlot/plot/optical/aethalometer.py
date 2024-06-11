# for Aethalometer instrument data

import matplotlib.pyplot as plt
import numpy as np
from pandas import date_range

from DataPlot.plot.core import set_figure


@set_figure(figsize=(10, 6))
def plot_MA350(df, **kwargs):
    fig, ax = plt.subplots(**kwargs.get('fig_kws', {}))

    # ax.scatter(df.index, df['UV BCc'], marker='o', c='purple', alpha=0.5, label='UV BCc')
    # ax.scatter(df.index, df['Blue BCc'], c='b', alpha=0.5, label='Blue BCc')
    # ax.scatter(df.index, df['Green BCc'], c='g', alpha=0.5, label='Green BCc')
    # ax.scatter(df.index, df['Red BCc'], c='r', alpha=0.5, label='Red BCc')
    mean, std = round(df.mean(), 2), round(df.std(), 2)

    label = rf'$IR\;BC:\;{mean["IR BCc"]}\;\pm\;{std["IR BCc"]}\;(ng/m^3)$'
    ax.plot(df.index, df['IR BCc'], ls='-', marker='o', c='k', alpha=0.5, label=label)
    ax.legend()

    st_tm, fn_tm = df.index[0], df.index[-1]
    tick_time = date_range(st_tm, fn_tm, freq=kwargs.get('freq', '1h'))

    ax.set(xlabel=kwargs.get('xlabel', ''),
           ylabel=kwargs.get('ylabel', r'$BC\ (ng/m^3)$'),
           xticks=kwargs.get('xticks', tick_time),
           xticklabels=kwargs.get('xticklabels', [_tm.strftime("%F %H:%M:%S") for _tm in tick_time]),
           xlim=kwargs.get('xlim', (st_tm, fn_tm)),
           ylim=kwargs.get('ylim', (0, None)),
           )


@set_figure
def plot_MA3502(df):
    fig, ax = plt.subplots()

    bins = np.array([375, 470, 528, 625, 880])
    vals = df.dropna().iloc[:, -5:].values

    ax.boxplot(vals, positions=bins, widths=20,
               showfliers=False, showmeans=True, meanline=True, patch_artist=True,
               boxprops=dict(facecolor='#f2c872', alpha=.7),
               meanprops=dict(color='#000000', ls='none'),
               medianprops=dict(ls='-', color='#000000'))

    ax.set(xlim=(355, 900),
           ylim=(0, None),
           xlabel=r'$\lambda\ (nm)$',
           ylabel=r'$Absorption\ (1/Mm)$', )


@set_figure
def plot_Bimass_Fossil(df):
    fig, ax = plt.subplots()
    ax.scatter(df.index, df['Biomass'], c='g', alpha=0.5, label='Biomass')
    ax.scatter(df.index, df['Fossil'], c='r', alpha=0.5, label='Fossil')
    ax.set(xlabel='Time', ylabel=r'$BC\ (ng/m^3)$')

    ax2 = ax.twinx()
    ax2.scatter(df.index, df['AAE'], c='b', alpha=0.5, label='AAE')
    ax2.set(ylabel='AAE', ylim=(0, 2))

    # 獲取ax的圖例句柄和標籤
    handles1, labels1 = ax.get_legend_handles_labels()

    # 獲取ax2的圖例句柄和標籤
    handles2, labels2 = ax2.get_legend_handles_labels()

    # 合併兩個圖例的句柄和標籤
    handles = handles1 + handles2
    labels = labels1 + labels2

    # 創建一個新的圖例
    ax.legend(handles, labels)

    return ax
