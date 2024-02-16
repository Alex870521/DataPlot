import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
from windrose import WindroseAxes
from DataPlot.process import *
from DataPlot.plot import StateClassifier, set_figure


@set_figure(fs=6)
def wind_heatmap(ws: pd.Series, wd: pd.Series, xticklabels):
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

    fig, ax = plt.subplots(figsize=(12, 6))
    uniform_data = [ws]
    colors = ['lightskyblue', 'darkturquoise', 'lime', 'greenyellow', 'orangered', 'red']
    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)  # 自定义色标
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

    plt.show()


@set_figure(fs=15)
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


if __name__ == "__main__":
    df = DataBase.copy()[:20]
    wind_heatmap(df['WS'], df['WD'], df.index.strftime('%F'))
    wind_rose(df['WS'], df['WD'])
