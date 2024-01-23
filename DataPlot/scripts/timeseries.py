import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import DataFrame, concat, date_range
from DataPlot.data_processing import *
from DataPlot.plot import *


@set_figure(fs=10)
def time_series(df):
    time = df.index.copy()
    st_tm, fn_tm = time[0], time[-1]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(len(df.index) * 0.02, 6))

    timeseries(df,
               y='Extinction',
               ax=ax1,
               plot_kws=dict(color="b", label='Extinction'),
               )

    timeseries(df,
               y='Scattering',
               ax=ax1,
               plot_kws=dict(color="g", label='Scattering'),
               )

    timeseries(df,
               y='Absorption',
               ax=ax1,
               plot_kws=dict(color="r", label='Absorption'),
               ylabel=r'$\bf Optical\ (1/Mm)$',
               ylim=[0., df.Extinction.max() * 1.1]
               )

    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, labelspacing=0.5, handlelength=1)

    # Temp, RH
    timeseries(df,
               y='AT',
               ax=ax2,
               plot_kws=dict(color='r', label=unit('AT')),
               ylabel=unit('AT'),
               ylim=[df.AT.min() - 2, df.AT.max() + 2]
               )

    timeseries(df,
               y='RH',
               ax=ax2.twinx(),
               plot_kws=dict(color='b', label=unit('RH')),
               ylabel=unit('RH'),
               ylim=[20, 100]
               )

    tms_bar(df,
            y='VC',
            color='PBLH',
            ax=ax3,
            plot_kws=dict(color='b', label=unit('VC')),
            ylabel=unit('VC'),
            ylim=[0, df.VC.max() * 1.1],
            cbar=True,
            cbar_kws=dict(ticks=[0, 400, 800], label=unit('PBLH'))
            )

    timeseries(df,
               y='WS',
               c='WD',
               ax=ax4,
               plot_kws=dict(cmap='hsv', label=unit('WS')),
               ylabel=unit('WS'),
               ylim=[0, df.WS.max() * 1.1],
               cbar=True,
               cbar_kws=dict(ticks=[0, 180, 360], label=unit('WD'))
               )

    timeseries(df,
               y='PM25',
               c='PM1/PM25',
               ax=ax5,
               set_visible=True,
               plot_kws=dict(vmin=0.2, vmax=1, cmap='jet', label=unit('PM25')),
               ylabel=unit('PM25'),
               cbar=True,
               cbar_kws=dict(label=unit('PM1/PM25'))
               )

    # fig.savefig(f'time2_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
    plt.show()
