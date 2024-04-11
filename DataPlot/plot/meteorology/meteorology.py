import matplotlib.colors as plc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from windrose import WindroseAxes

from DataPlot.plot.core import *
from DataPlot.process import *

__all__ = ['wind_tms',
           'wind_rose',
           'wind_heatmap',
           ]


@set_figure(fs=6)
def wind_tms(df: pd.DataFrame, ws: pd.Series, wd: pd.Series, xticklabels):
    def drawArrow(A, B, ax: plt.Axes):  # 畫箭頭
        _ax = ax.twinx()
        if A[0] == B[0] and A[1] == B[1]:  # 靜風畫點
            _ax.plot(A[0], A[1], 'ko')
        else:
            _ax.annotate("", xy=(B[0], B[1]), xytext=(A[0], A[1]), arrowprops=dict(arrowstyle="->"))

        _ax.spines['left'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.spines['top'].set_visible(False)
        _ax.spines['bottom'].set_visible(False)
        _ax.set_xlim(0, )
        _ax.set_ylim(0, 5)
        _ax.get_yaxis().set_visible(False)
        _ax.set_aspect('equal')  # x轴y轴等比例

        _ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='x', rotation=90)
        plt.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 2))
    uniform_data = [ws]
    colors = ['lightskyblue', 'darkturquoise', 'lime', 'greenyellow', 'orangered', 'red']
    clrmap = plc.LinearSegmentedColormap.from_list("mycmap", colors)  # 自定义色标
    sns.heatmap(uniform_data, square=True, annot=True, fmt=".2f", linewidths=.5, cmap=clrmap,
                yticklabels=['Wind speed (m/s)'], xticklabels=xticklabels, cbar=False, vmin=0, vmax=5, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.spines['bottom'].set_position(('data', 1))  # 移动x轴

    for idx, (x, value) in enumerate(wd.items()):
        if not pd.isna(value):
            a = np.array([0.5 + 0.5 * np.sin(value / 180 * np.pi) + idx, 3.5 + 0.5 * np.cos(value / 180 * np.pi)])
            b = np.array([0.5 - 0.5 * np.sin(value / 180 * np.pi) + idx, 3.5 - 0.5 * np.cos(value / 180 * np.pi)])
            drawArrow(a, b, ax)
        else:
            a = np.array([0.5 + idx, 3.5])
            drawArrow(a, a, ax)

    return ax


@set_figure
def wind_rose(ws: pd.Series, wd: pd.Series):
    color_lst = ['#8ecae6', '#f1dca7', '#f4a261', '#bc3908']

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')

    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(wd.values, ws.values, bins=[0, 1, 2, 3], nsector=16, normed=True, colors=color_lst)

    ax.set(ylim=(0, 30), yticks=[0, 15, 30], yticklabels=['', '15 %', '30 %'])
    ax.tick_params(pad=-5)
    plt.show()
    # ax.set_legend(framealpha=0, bbox_to_anchor=[-.05,-.05], fontsize=fs-2., loc='lower left', ncol=3)
    # fig.savefig(f'windrose/windrose_{state}.png')

    return ax


def wind_heatmap(ws, wd, values):  # CBPF
    # TODO:
    ws = ws.to_numpy()
    wd = wd.to_numpy()
    values = values.to_numpy()

    x = []
    y = []
    val = []
    theta = []
    for _ws, _wd, _values in zip(ws, wd, values):
        if not pd.isna(_values):
            x.append(_ws*np.sin(_wd/180 * np.pi))
            y.append(_ws*np.cos(_wd/180 * np.pi))
            val.append(_ws)
            theta.append(_wd/180 * np.pi)
        else:
            pass

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # plt.scatter(theta, val)

    # surf, _ = ax.pcolormesh(x, y, val, shading='auto', antialiased=True)
    pass


if __name__ == "__main__":
    df = DataBase.copy()
    # wind_heatmap(df, df['WS'], df['WD'], df.index.strftime('%F'))
    # wind_rose(df['WS'], df['WD'])
    df = df[['WS', 'WD', 'Extinction']].dropna()
    wind_heatmap(df['WS'], df['WD'], df['Extinction'])
