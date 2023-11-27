import numpy as np
import math
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
# from PSD_extinction import extinction_psd_process
from DataPlot.Data_processing.PSD_reader import psd_reader


class DataTypeError(Exception):
    """ make sure the input data unit is dN/dlogdp """


class SizeDist:  # 可以加入一些錯誤的raise
    """
    A class for processing particle size distribution (PSD) data.

    # Examples
    --------
    Example 1: Use default path and filename
    >>> psd_data = SizeDist()

    Example 2: Specify custom path and filename
    >>> custom_psd_data = SizeDist(path='custom/path', filename='custom_PSD.csv')

    Parameters
    ----------
    path : Path, optional
        The directory path where the PSD data file is located.
    filename : str, optional
        The name of the PSD data file.
        Defaults to 'PNSD_dNdlogdp.csv' in the default path.

    Attributes
    ----------
    path : Path
        The directory path where the PSD data file is located.
    filename : str
        The name of the PSD data file.
    data : DataFrame
        The processed PSD data stored as a pandas DataFrame.
    index : DatetimeIndex
        The index of the DataFrame representing time.
    dp : ndarray
        The array of particle diameters from the PSD data.
    dlogdp : ndarray
        The array of logarithmic particle diameter bin widths.

    Methods
    -------
    number()
        Calculate number distribution properties.

    surface(filename='PSSD_dSdlogdp.csv')
        Calculate surface distribution properties.

    volume(filename='PVSD_dVdlogdp.csv')
        Calculate volume distribution properties.

    extinction(filename='PESD_dEdlogdp.csv')
        Placeholder for extinction distribution properties.

    psd_process(filename='PSD.csv')
        Process and save overall PSD properties.

    """

    default_path = Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'distribution' / 'PNSD_dNdlogdp.csv'

    def __init__(self, path=None, filename=None):
        self.path = path or self.default_path.parent
        self.filename = filename or self.default_path.name
        self.data = psd_reader(self.path / self.filename).dropna()
        self.index = self.data.index.copy()
        self.dp = np.array(self.data.columns, dtype='float')
        self.dlogdp = np.full_like(self.dp, 0.014)

    def number(self):
        num_dist = self.data
        num_prop = num_dist.apply(self.__dist_prop, axis=1, result_type='expand')

        return pd.DataFrame({'Number': num_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDn': num_prop['GMD'],
                             'GSDn': num_prop['GSD'],
                             'mode_n': num_prop['mode'],
                             'cont_n': num_prop['contribution']})

    def surface(self, filename='PSSD_dSdlogdp.csv'):
        surf_dist = self.data.apply(lambda col: math.pi * (self.dp ** 2) * np.array(col), axis=1, result_type='broadcast')
        surf_prop = surf_dist.apply(self.__dist_prop, axis=1, result_type='expand')

        surf_dist.reindex(self.index).to_csv(self.path / filename)

        return pd.DataFrame({'Surface': surf_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDs': surf_prop['GMD'],
                             'GSDs': surf_prop['GSD'],
                             'mode_s': surf_prop['mode'],
                             'cont_s': surf_prop['contribution']})

    def volume(self, filename='PVSD_dVdlogdp.csv'):
        vol_dist = self.data.apply(lambda col: math.pi / 6 * self.dp ** 3 * np.array(col), axis=1, result_type='broadcast')
        vol_prop = vol_dist.apply(self.__dist_prop, axis=1, result_type='expand')

        vol_dist.reindex(self.index).to_csv(self.path / filename)

        return pd.DataFrame({'Volume': vol_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDv': vol_prop['GMD'],
                             'GSDv': vol_prop['GSD'],
                             'mode_v': vol_prop['mode'],
                             'cont_v': vol_prop['contribution']})

    def extinction(self, filename='PESD_dEdlogdp.csv'):
        pass

    def psd_process(self, filename='PSD.csv'):
        return pd.concat([self.number(), self.surface(), self.volume()], axis=1).reindex(self.index).to_csv(self.path.parent / filename)

    def __geometric_prop(self, ser):
        """ First change the distribution into dN """
        num = np.array(ser) * self.dlogdp
        total_num = num.sum()

        _dp = np.log(self.dp)
        _gmd = (((num * _dp).sum()) / total_num.copy())

        _dp_mesh, _gmd_mesh = np.meshgrid(_dp, _gmd)
        _gsd = ((((_dp_mesh - _gmd_mesh) ** 2) * num).sum() / total_num.copy()) ** .5

        return np.exp(_gmd), np.exp(_gsd)

    def __mode_prop(self, ser):
        min_value = np.array([min(ser)])
        extend_ser = np.concatenate([min_value, ser, min_value])
        _mode, _ = find_peaks(extend_ser, distance=20)
        return self.dp[_mode - 1]

    def __mode_contribution(self, ser):
        num = np.array(ser) * self.dlogdp
        total_num = num.sum()

        ultra_range = (self.dp >= 11.8) & (self.dp < 100)
        accum_range = (self.dp >= 100) & (self.dp < 1000)
        coarse_range = (self.dp >= 1000) & (self.dp < 2500)

        ultra_num = np.sum(num[ultra_range])
        accum_num = np.sum(num[accum_range])
        coars_num = np.sum(num[coarse_range])

        return [(ultra_num / total_num), (accum_num / total_num), (coars_num / total_num)]

    def __dist_prop(self, ser):
        GMD, GSD = self.__geometric_prop(ser)
        Mode = self.__mode_prop(ser)
        contribution = self.__mode_contribution(ser)

        return dict(GMD=GMD, GSD=GSD, mode=Mode, contribution=contribution, )


if __name__ == '__main__':
    PNSD_data = SizeDist()

    # file_path = Path('C:/Users/Alex/PycharmProjects/DataPlot/Data/Level2/distribution')
    # PNSD_data = SizeDist(path=file_path, filename='PNSD_dNdlogdp_dry.csv')




