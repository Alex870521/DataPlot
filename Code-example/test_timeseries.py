import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy.random import multivariate_normal

import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot import set_figure, unit, getColor, color_maker
from pandas import DataFrame, concat, date_range, Timestamp
from DataPlot.scripts import *
from DataPlot.data_processing import *


if __name__ == '__main__':
    PNSD = DataReader('PNSD_dNdlogdp.csv')
    PSSD = DataReader('PSSD_dSdlogdp.csv')
    PVSD = DataReader('PVSD_dVdlogdp.csv')
    PESD = DataReader('PESD_dextdlogdp_internal.csv')
    df   = DataReader('All_data.csv')


    plt.style.use('_mpl-gallery-nogrid')
    PNSD=PVSD.dropna(axis=0)
    # plot:
    fig, ax = plt.subplots()
    original_array = np.array(PNSD.columns)
    x = np.repeat(original_array, len(PNSD.index)).astype(float)
    y = PNSD.values.flatten().astype(float)
    ax.hist2d(x, y, bins=167, norm=mcolors.PowerNorm(0.3))

    plt.show()


    # Season timeseries
    for season, (st_tm_, fn_tm_) in Seasons.items():
        st_tm, fn_tm = pd.Timestamp(st_tm_), pd.Timestamp(fn_tm_)

        df = df.loc[st_tm:fn_tm].copy()

        PNSD_data = PNSD.loc[st_tm:fn_tm]
        PSSD_data = PSSD.loc[st_tm:fn_tm]

        # 數據平滑
        df = df.rolling(3).mean(numeric_only=True)

        # time_series(df)

        break
