import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pandas import DataFrame, concat, date_range
from DataPlot.process import *
from DataPlot.plot import set_figure, Unit, timeseries


@set_figure(fs=10)
def time_series(df):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(len(df.index) * 0.02, 6))

    ax1 = timeseries(df,
                     y='Extinction',
                     ax=ax1,
                     set_visible=False,
                     plot_kws=dict(color="b", label='Extinction'),
                     ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                     ylim=[0., df.Extinction.max() * 1.1]
                     )

    ax1 = timeseries(df,
                     y='Scattering',
                     ax=ax1,
                     set_visible=False,
                     plot_kws=dict(color="g", label='Scattering'),
                     ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                     ylim=[0., df.Extinction.max() * 1.1]
                     )

    ax1 = timeseries(df,
                     y='Absorption',
                     ax=ax1,
                     set_visible=False,
                     plot_kws=dict(color="r", label='Absorption'),
                     ylabel=r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$',
                     ylim=[0., df.Extinction.max() * 1.1]
                     )

    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, labelspacing=0.5, handlelength=1)

    # Temp, RH
    ax2 = timeseries(df,
                     y='AT',
                     ax=ax2,
                     set_visible=False,
                     plot_kws=dict(color='r', label=Unit('AT')),
                     ylabel=Unit('AT'),
                     ylim=[df.AT.min() - 2, df.AT.max() + 2])

    timeseries(df,
               y='RH',
               ax=ax2.twinx(),
               set_visible=False,
               plot_kws=dict(color='b', label=Unit('RH')),
               ylabel=Unit('RH'),
               ylim=[20, 100])

    timeseries(df,
               y='VC',
               c='PBLH',
               style='bar',
               ax=ax3,
               set_visible=False,
               plot_kws=dict(label=Unit('VC')),
               cbar_kws=dict(label=Unit('PBLH'))
               )

    timeseries(df,
               y='WS',
               c='WD',
               ax=ax4,
               set_visible=False,
               plot_kws=dict(cmap='hsv', label=Unit('WS')),
               cbar_kws=dict(label=Unit('WD')),
               ylim=[0, df.WS.max() * 1.1]
               )

    timeseries(df,
               y='PM25',
               c='PM1/PM25',
               ax=ax5,
               plot_kws=dict(label=Unit('PM1/PM25')),
               cbar_kws=dict(label=Unit('PM1/PM25')),
               ylim=[0, df.PM25.max() * 1.1]
               )

    # fig.savefig(f'time2_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
    plt.show()


if __name__ == '__main__':
    time_series(DataBase[:720])
