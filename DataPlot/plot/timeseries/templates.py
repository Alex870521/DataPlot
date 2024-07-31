import matplotlib.pyplot as plt
from pandas import DataFrame

from DataPlot.plot.core import *
from DataPlot.plot.timeseries.timeseries import timeseries


@set_figure(fs=8, autolayout=False)
def timeseries_template(df: DataFrame):
    fig, ax = plt.subplots(5, 1, figsize=(len(df.index) * 0.01, 4))
    (ax1, ax2, ax3, ax4, ax5) = ax

    timeseries(df,
               y=['Extinction', 'Scattering', 'Absorption'],
               rolling=30,
               ax=ax1,
               ylabel='Coefficient',
               ylim=[0., None],
               set_xaxis_visible=False,
               legend_ncol=3,
               )

    # Temp, RH
    timeseries(df,
               y='AT',
               y2='RH',
               rolling=30,
               ax=ax2,
               ax_plot_kws=dict(color='r'),
               ax2_plot_kws=dict(color='b'),
               ylim=[10, 40],
               ylim2=[20, 100],
               set_xaxis_visible=False,
               legend_ncol=2,
               )

    timeseries(df, y='WS', c='WD', ax=ax3, scatter_kws=dict(cmap='hsv'), cbar_kws=dict(ticks=[0, 90, 180, 270, 360]),
               ylim=[0, None], set_xaxis_visible=False)

    timeseries(df, y='VC', c='PBLH', style='bar', ax=ax4, bar_kws=dict(cmap='Blues'), set_xaxis_visible=False,
               ylim=[0, 5000])

    timeseries(df, y='PM25', c='PM1/PM25', ax=ax5, ylim=[0, None])


if __name__ == '__main__':
    from DataPlot import DataBase

    df = DataBase('/Users/chanchihyu/NTU/2020能見度計畫/data/All_data.csv')
    timeseries_template(df.loc['2020-09-01':'2020-12-31'])
