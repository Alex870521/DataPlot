import numpy as np
import math
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from PSD_extinction import extinction_psd_process
from PSD_reader import _reader
from DataPlot.Data_processing.csv_decorator import save_to_csv


class DataTypeError(Exception):
    """ make sure the input data unit is dN/dlogdp """


class SizeDist:
    """ 輸入粒徑分布檔案，計算表面機、體積分布與幾何平均粒徑等方法 """
    path_main = Path(__file__).parent.parent.parent / 'Data' / 'Level2'
    path_dist = path_main / 'distribution'

    def __init__(self, df):
        self.dp = np.array(df.columns, dtype='float')
        self.dlogdp = np.array([0.014] * np.size(self.dp))
        self.index = df.index.copy()
        self.data = df.dropna()

    def number(self, reset=False, **kwargs):
        num_dist = self.data
        num_prop = num_dist.apply(self.__geometric_prop, axis=1, result_type='expand')

        return pd.DataFrame({'Number': num_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDn': num_prop[0],
                             'GSDn': num_prop[1]})

    def surface(self, reset=False, filename='PSSD_dSdlogdp.csv', **kwargs):
        surf_dist = self.data.apply(lambda col: math.pi * (self.dp ** 2) * np.array(col), axis=1, result_type='broadcast')
        surf_prop = surf_dist.apply(self.__geometric_prop, axis=1, result_type='expand')

        surf_dist.reindex(self.index).to_csv(self.path_dist / filename)

        return pd.DataFrame({'Surface': surf_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDs': surf_prop[0],
                             'GSDs': surf_prop[1]})

    def volume(self, reset=False, filename='PVSD_dVdlogdp.csv', **kwargs):
        vol_dist = self.data.apply(lambda col: math.pi / 6 * self.dp ** 3 * np.array(col), axis=1, result_type='broadcast')
        vol_prop = vol_dist.apply(self.__geometric_prop, axis=1, result_type='expand')

        vol_dist.reindex(self.index).to_csv(self.path_dist / filename)

        return pd.DataFrame({'Volume': vol_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDv': vol_prop[0],
                             'GSDv': vol_prop[1]})

    def extinction(self):
        pass
        # return extinction_psd_process(data=self.data, reset=True)

    def export_to_csv(self, filename='PSD.csv'):
        num_df = self.number()
        surf_df = self.surface()
        vol_df = self.volume()

        result_df = pd.concat([num_df, surf_df, vol_df], axis=1)

        result_df.reindex(self.index).to_csv(self.path_main / filename)

    def __geometric_prop(self, ser):
        num = np.array(ser)
        total_num = num.sum()

        _dp = np.log(self.dp)
        _gmd = (((num * _dp).sum()) / total_num.copy())

        _dp_mesh, _gmd_mesh = np.meshgrid(_dp, _gmd)
        _gsd = ((((_dp_mesh - _gmd_mesh) ** 2) * num).sum() / total_num.copy()) ** .5

        return np.exp(_gmd), np.exp(_gsd)

    def __dist_prop(self, dist):
        peaks1, _ = find_peaks(np.concatenate(([min(dist)], dist, [min(dist)])), distance=20)
        num = np.array(dist * self.dlogdp)
        total_num = np.sum(num)

        ultra_num = np.sum(num[0:67])
        accum_num = np.sum(num[67:139])
        PM1_num = (ultra_num + accum_num)
        coars_num = np.sum(num[139:167])

        GMD, GSD = self.__geometric_prop(self.dp, num)

        contrbution = [(ultra_num / total_num), (accum_num / total_num), (coars_num / total_num)]
        return dict(mode=self.dp[peaks1 - 1], GMD=GMD, GSD=GSD, PM1_num=PM1_num, PM25_num=total_num,
                    contrbution=contrbution, )


if __name__ == '__main__':
    PNSD_data = SizeDist(_reader())
    PNSD_data.export_to_csv()
