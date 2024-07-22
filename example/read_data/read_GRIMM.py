from datetime import datetime as dtm
from pathlib import Path

import pandas as pd

from DataPlot import *

start, end = dtm(2024, 4, 25), dtm(2024, 5, 10)

path_raw = Path('/Users/chanchihyu/NTU/監資司資料/GRIMM_data/raw')
path_prcs = Path('prcs')


def plot_dist(data: pd.DataFrame):
    plot.distribution.heatmap(data, unit='Number', magic_number=0)
    plot.distribution.heatmap_tms(data, unit='Number', freq='10d')


if __name__ == '__main__':
    df = RawDataReader('GRIMM', path_raw / 'A407ST', reset=True, start=start, end=end, mean_freq='1h', csv_out=True)
    df2 = RawDataReader('GRIMM', path_raw / 'A812SK', reset=True, start=start, end=end, mean_freq='1h', csv_out=True)
    df3 = RawDataReader('GRIMM', path_raw / 'AQMV12', reset=True, start=start, end=end, mean_freq='1h', csv_out=True)

    plot_dist(df)
    plot_dist(df2)
    plot_dist(df3)
