import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_csv, concat
from PSD_surface_volume import Number_PSD_process, Surface_PSD_process, Volume_PSD_process
from PSD_extinction import Extinction_PSD_process


class SizeDist:
    """ 輸入粒徑分布，計算表面機、體積分布與幾何平均粒徑等方法 """  # == def __doc__()

    def __init__(self, df):
        self.dp = np.array(df.columns, dtype='float')
        self.dlogdp = np.array([0.014] * np.size(self.dp))
        self.index = df.index.copy()
        self.data = df.dropna()


    @staticmethod
    def method(self):
        print('Hellow')

    @property
    def number(self):
        return Number_PSD_process(data=self.data, reset=True)

    def surface(self):
        return Surface_PSD_process(data=self.data, reset=True)

    def volume(self):
        return Volume_PSD_process(data=self.data, reset=True)

    def extinction(self):
        return Extinction_PSD_process(data=self.data, reset=True)


if __name__ == '__main__':
    PATH_DIST = Path("C:/Users/Alex/PycharmProjects/DataPlot/Data/Level2/distribution")

    with open(PATH_DIST / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    PNSD_data = SizeDist('main', PNSD)

    sur = PNSD_data.surface

