import math
from typing import Literal

import matplotlib.colors as plc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series
from scipy.ndimage import gaussian_filter

from DataPlot.plot.core import *
from DataPlot.process import *

__all__ = ['wind_tms',
           'wind_rose',
           ]


@set_figure(fs=6)
def wind_tms(df: DataFrame,
             ws: Series,
             wd: Series,
             **kwargs
             ):
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
                yticklabels=['Wind speed (m/s)'], xticklabels=kwargs.get('xticklabels', None), cbar=False, vmin=0,
                vmax=5, ax=ax)
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


@set_figure(figsize=(4.3, 4))
def wind_rose(df: DataFrame,
              WS: Series | str,
              WD: Series | str,
              val: Series | str | None = None,
              typ: Literal['bar', 'scatter', 'cbpf'] = 'cbpf',
              percentile: list | float | int | None = None,
              masked_less: bool = 0.05,
              max_ws: float | None = 5,
              resolution: int = 25,
              sigma: float | tuple = 5,
              rlabel_pos: float = 30,
              **kwargs
              ):
    # conditional bivariate probability function (cbpf) python
    # https://davidcarslaw.github.io/openair/reference/polarPlot.html
    # https://github.com/davidcarslaw/openair/blob/master/R/polarPlot.R

    df = df.dropna(subset=[WS, WD] + ([val] if val is not None else []))

    radius = df[WS].to_numpy()
    theta = df[WD].to_numpy()
    radian = np.radians(theta)
    values = df[val].to_numpy() if val is not None else None

    # In this case, the windrose is a simple frequency diagram,
    # the function automatically calculates the radians of the given wind direction.
    if typ == 'bar':
        fig, ax = plt.subplots(figsize=(5.5, 4), subplot_kw={'projection': 'windrose'})
        fig.subplots_adjust(left=0)

        ax.bar(theta, radius, bins=[0, 1, 2, 3], normed=True, colors=['#0F1035', '#365486', '#7FC7D9', '#DCF2F1'])
        ax.set(
            ylim=(0, 30),
            yticks=[0, 15, 30],
            yticklabels=['', '15 %', '30 %'],
            rlabel_position=rlabel_pos
        )
        ax.set_thetagrids(angles=[0, 45, 90, 135, 180, 225, 270, 315],
                          labels=["E", "NE", "N", "NW", "W", "SW", "S", "SE"])

        ax.legend(units='m/s', bbox_to_anchor=[1.1, 0.5], loc='center left', ncol=1)

    # In this case, the windrose is a scatter plot,
    # in contrary, this function does not calculate the radians, so user have to input the radian.
    elif typ == 'scatter':
        fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={'projection': 'windrose'})
        fig.subplots_adjust(left=0)

        scatter = ax.scatter(radian, radius, s=20, c=values, vmax=np.quantile(values, 0.90), edgecolors='none',
                             cmap='jet', alpha=0.5)
        ax.set(
            ylim=(0, 7),
            yticks=[1, 3, 5, 7],
            yticklabels=['1 m/s', '3 m/s', '5 m/s', '7 m/s'],
            rlabel_position=rlabel_pos,
            theta_direction=-1,
            theta_zero_location='N',
        )
        ax.set_thetagrids(angles=[0, 45, 90, 135, 180, 225, 270, 315],
                          labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        plt.colorbar(scatter, ax=ax, label=Unit(val), pad=0.1, fraction=0.04)

    # The Bivariate probability function (bpf) plot in Cartesian coordinate.
    else:
        df = df.copy()
        df['u'] = df[WS].to_numpy() * np.sin(np.radians(df[WD].to_numpy()))
        df['v'] = df[WS].to_numpy() * np.cos(np.radians(df[WD].to_numpy()))

        u_bins = np.arange(df.u.min(), df.u.max(), 1 / resolution)
        v_bins = np.arange(df.v.min(), df.v.max(), 1 / resolution)

        df['u_group'] = pd.cut(df['u'], u_bins)
        df['v_group'] = pd.cut(df['v'], v_bins)

        # 使用 u_group 和 v_group 進行分組
        grouped = df.groupby(['u_group', 'v_group'], observed=False)

        if percentile is None:
            histogram, v_edges, u_edges = np.histogram2d(df.v, df.u, bins=(v_bins, u_bins))
            X, Y = np.meshgrid(u_bins, v_bins)
            bottom_text = None

        else:
            if not all(0 <= p <= 100 for p in (percentile if isinstance(percentile, list) else [percentile])):
                raise ValueError("Percentile must be between 0 and 100")

            if isinstance(percentile, (float, int)):
                bottom_text = rf'$CPF:\ >{int(percentile)}^{{th}}$'
                thershold = df[val].quantile(percentile / 100)
                cond = lambda x: (x >= thershold).sum()

            else:
                bottom_text = rf'$CPF:\ {int(percentile[0])}^{{th}}\ to\ {int(percentile[1])}^{{th}}$'
                thershold_small, thershold_large = df[val].quantile([percentile[0] / 100, percentile[1] / 100])
                cond = lambda x: ((x >= thershold_small) & (x < thershold_large)).sum()

            histogram = (grouped[val].apply(cond) / grouped[val].count()).unstack().values
            histogram = np.nan_to_num(histogram, nan=0.0001).T
            # histogram = np.ma.masked_invalid(histogram).T

            X, Y = np.meshgrid(u_bins, v_bins)

        if masked_less:
            masked_array = np.ma.masked_less(gaussian_filter(histogram, sigma=sigma), masked_less)
            vmax = np.percentile(np.ma.compressed(masked_array), 99.5)

        else:
            masked_array = gaussian_filter(histogram, sigma=sigma)
            vmax = np.percentile(masked_array, 99.5)

        # plot
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0)

        surf = ax.pcolormesh(X, Y, masked_array, shading='auto', vmax=vmax, cmap='jet', antialiased=True)

        max_ws = max_ws or np.concatenate((abs(df.u), abs(df.v))).max()  # Get the maximum value of the wind speed

        radius_lst = np.arange(1, math.ceil(max_ws) + 1)  # Create a list of radius

        for i, radius in enumerate(radius_lst):
            circle = plt.Circle((0, 0), radius, fill=False, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            ax.add_artist(circle)

            for angle, label in zip(range(0, 360, 90), ["E", "N", "W", "S"]):
                radian = np.radians(angle)
                line_x, line_y = radius * np.cos(radian), radius * np.sin(radian)

                if i + 2 == len(radius_lst):  # Add wind direction line and direction label at the edge of the circle
                    ax.plot([0, line_x * 1.05], [0, line_y * 1.05], color='k', linestyle='-', linewidth=1, alpha=0.5)
                    ax.text(line_x * 1.15, line_y * 1.15, label, ha='center', va='center')

            ax.text(radius * np.cos(np.radians(rlabel_pos)), radius * np.sin(np.radians(rlabel_pos)),
                    str(radius) + ' m/s', ha='center', va='center', fontsize=8)

        for radius in range(math.ceil(max_ws) + 1, 10):
            circle = plt.Circle((0, 0), radius, fill=False, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            ax.add_artist(circle)

        ax.set(xlim=(-max_ws * 1.02, max_ws * 1.02),
               ylim=(-max_ws * 1.02, max_ws * 1.02),
               xticks=[],
               yticks=[],
               xticklabels=[],
               yticklabels=[],
               aspect='equal',
               )

        ax.text(0.50, -0.05, bottom_text, fontweight='bold', fontsize=8, va='center', ha='center',
                transform=ax.transAxes)
        ax.text(0.03, 0.97, Unit(val), fontweight='bold', fontsize=12, va='top', ha='left', transform=ax.transAxes)

        cbar = plt.colorbar(surf, ax=ax, label='Frequency', pad=0.01, fraction=0.04)
        cbar.ax.yaxis.label.set_fontsize(8)
        cbar.ax.tick_params(labelsize=8)


if __name__ == "__main__":
    df = DataBase().copy()
    df1 = df[['WS', 'WD', 'PM25', 'NO2', 'O3']]

    wind_rose(df, 'WS', 'WD', typ='bar')
    wind_rose(df, 'WS', 'WD', 'PM25', typ='scatter')
    wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf')
    # wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf', percentile=[0, 25])
    # wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf', percentile=[25, 50])
    # wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf', percentile=[50, 75])
    # wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf', percentile=[75, 100])
