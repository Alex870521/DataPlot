import numpy as np
import math
import pandas as pd
from pathlib import Path
from pandas import read_csv, concat
from scipy.signal import find_peaks
from PSD_surface_volume import number_psd_process, surface_psd_process, volume_psd_process
from PSD_extinction import extinction_psd_process
from PSD_reader import _reader


class DataTypeError(Exception):
    """ make sure the input data unit is dN/dlogdp """


class SizeDist:
    """ 輸入粒徑分布，計算表面機、體積分布與幾何平均粒徑等方法 """

    def __init__(self, df):
        self.dp = np.array(df.columns, dtype='float')
        self.dlogdp = np.array([0.014] * np.size(self.dp))
        self.index = df.index.copy()
        self.data = df.dropna()

    @property
    def number(self):
        return number_psd_process(data=self.data, reset=True)

    def surface(self):
        return surface_psd_process(data=self.data, reset=True)

    def volume(self):
        return volume_psd_process(data=self.data, reset=True)

    def extinction(self):
        return extinction_psd_process(data=self.data, reset=True)

    def geometric_prop(self, column):
        num = np.array(column)
        total_num = num.sum()

        _dp = np.log(self.dp)
        _gmd = (((num * _dp).sum()) / total_num.copy())

        _dp_mesh, _gmd_mesh = np.meshgrid(_dp, _gmd)
        _gsd = ((((_dp_mesh - _gmd_mesh) ** 2) * num).sum() / total_num.copy()) ** .5

        return np.exp(_gmd), np.exp(_gsd)

    def dist_prop(self, dist):
        peaks1, _ = find_peaks(np.concatenate(([min(dist)], dist, [min(dist)])), distance=20)
        num = np.array(dist * self.dlogdp)
        total_num = np.sum(num)

        ultra_num = np.sum(num[0:67]).__round__(4)
        accum_num = np.sum(num[67:139]).__round__(4)
        PM1_num = (ultra_num + accum_num).__round__(4)
        coars_num = np.sum(num[139:167]).__round__(4)

        GMD, GSD = self.geometric_prop(self.dp, num)

        contrbution = [(ultra_num / total_num).__round__(4), (accum_num / total_num).__round__(4),
                       (coars_num / total_num).__round__(4)]
        return dict(mode=self.dp[peaks1 - 1], GMD=GMD, GSD=GSD, PM1_num=PM1_num, PM25_num=total_num,
                    contrbution=contrbution, )


if __name__ == '__main__':
    PNSD_data = SizeDist(_reader())

    sur = PNSD_data.surface

