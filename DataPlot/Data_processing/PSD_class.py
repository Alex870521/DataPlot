import numpy as np
import math
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from DataPlot.Data_processing.csv_decorator import save_to_csv
from DataPlot.Data_processing import psd_reader, chemical_reader
from DataPlot.Data_processing.Mie_plus import Mie_PESD


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
    >>> custom_psd_data = SizeDist(path=Path('custom/path'), filename='custom_PSD.csv')

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
        """ dN/dlogdp """
        num_dist = self.data
        num_prop = num_dist.apply(self.__dist_prop, axis=1, result_type='expand')

        return pd.DataFrame({'Number': num_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDn': num_prop['GMD'],
                             'GSDn': num_prop['GSD'],
                             'mode_n': num_prop['mode'],
                             'cont_n': num_prop['contribution']})

    def surface(self, filename='PSSD_dSdlogdp.csv'):
        surf_dist = self.data.apply(lambda col: math.pi * (self.dp ** 2) * np.array(col), axis=1,
                                    result_type='broadcast')
        surf_prop = surf_dist.apply(self.__dist_prop, axis=1, result_type='expand')

        surf_dist.reindex(self.index).to_csv(self.path / filename)

        return pd.DataFrame({'Surface': surf_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDs': surf_prop['GMD'],
                             'GSDs': surf_prop['GSD'],
                             'mode_s': surf_prop['mode'],
                             'cont_s': surf_prop['contribution']})

    def volume(self, filename='PVSD_dVdlogdp.csv'):
        vol_dist = self.data.apply(lambda col: math.pi / 6 * self.dp ** 3 * np.array(col), axis=1,
                                   result_type='broadcast')
        vol_prop = vol_dist.apply(self.__dist_prop, axis=1, result_type='expand')

        vol_dist.reindex(self.index).to_csv(self.path / filename)

        return pd.DataFrame({'Volume': vol_dist.apply(np.sum, axis=1) * 0.014,
                             'GMDv': vol_prop['GMD'],
                             'GSDv': vol_prop['GSD'],
                             'mode_v': vol_prop['mode'],
                             'cont_v': vol_prop['contribution']})

    def extinction_internal(self, filename='PESD_dextdlogdp_internal.csv'):
        ext_data = pd.concat([self.data, chemical_reader()], axis=1).dropna()

        result_dic = ext_data.apply(self.__internal_ext_dist, axis=1, result_type='expand')

        ext_dist = pd.DataFrame(result_dic['ext'].tolist(), index=result_dic['ext'].index).set_axis(self.dp, axis=1)
        sca_dist = pd.DataFrame(result_dic['sca'].tolist(), index=result_dic['sca'].index).set_axis(self.dp, axis=1)
        abs_dist = pd.DataFrame(result_dic['abs'].tolist(), index=result_dic['abs'].index).set_axis(self.dp, axis=1)

        ext_prop = ext_dist.apply(self.__dist_prop, axis=1, result_type='expand')

        ext_dist.reindex(self.index).to_csv(self.path / filename)

        return pd.DataFrame({'Bext_internal': ext_dist.apply(np.sum, axis=1) * 0.014,
                             'Bsca_internal': sca_dist.apply(np.sum, axis=1) * 0.014,
                             'Babs_internal': abs_dist.apply(np.sum, axis=1) * 0.014,
                             'GMD_ext_in': ext_prop['GMD'],
                             'GSD_ext_in': ext_prop['GSD'],
                             'mode_ext_in': ext_prop['mode'], })

    def extinction_external(self, filename='PESD_dextdlogdp_external.csv'):
        ext_data = pd.concat([self.data, chemical_reader()], axis=1).dropna()

        result_dic2 = ext_data.apply(self.__external_ext_dist, axis=1, result_type='expand')

        ext_dist2 = pd.DataFrame(result_dic2['ext'].tolist(), index=result_dic2['ext'].index).set_axis(self.dp, axis=1)
        sca_dist2 = pd.DataFrame(result_dic2['sca'].tolist(), index=result_dic2['sca'].index).set_axis(self.dp, axis=1)
        abs_dist2 = pd.DataFrame(result_dic2['abs'].tolist(), index=result_dic2['abs'].index).set_axis(self.dp, axis=1)

        ext_prop2 = ext_dist2.apply(self.__dist_prop, axis=1, result_type='expand')

        ext_dist2.reindex(self.index).to_csv(self.path / filename)

        return pd.DataFrame({'Bext_external': ext_dist2.apply(np.sum, axis=1) * 0.014,
                             'Bsca_external': sca_dist2.apply(np.sum, axis=1) * 0.014,
                             'Babs_external': abs_dist2.apply(np.sum, axis=1) * 0.014,
                             'GMD_ext_ex': ext_prop2['GMD'],
                             'GSD_ext_ex': ext_prop2['GSD'],
                             'mode_ext_ex': ext_prop2['mode'], })

    def psd_process(self, reset=None, filename='PSD.csv'):
        result_df = pd.concat([self.number(), self.surface(), self.volume()], axis=1).reindex(self.index)
        result_df.to_csv(self.path.parent / filename)
        return result_df

    def ext_process(self, reset=None, filename='PESD.csv'):
        result_df = pd.concat([self.extinction_internal(), self.extinction_external(), ], axis=1).reindex(self.index)
        result_df.to_csv(self.path.parent / filename)
        return result_df

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
        gmd, gsd = self.__geometric_prop(ser)
        mode = self.__mode_prop(ser)
        contribution = self.__mode_contribution(ser)

        return dict(GMD=gmd, GSD=gsd, mode=mode, contribution=contribution, )

    def __internal_ext_dist(self, ser):
        m = ser['n_amb'] + 1j * ser['k_amb']
        ndp = np.array(ser[:np.size(self.dp)])
        ext_dist, sca_dist, abs_dist = Mie_PESD(m, 550, self.dp, self.dlogdp, ndp)
        return dict(ext=ext_dist, sca=sca_dist, abs=abs_dist)

    def __external_ext_dist(self, ser):
        refractive_dic = {'AS': 1.53 + 0j,
                          'AN': 1.55 + 0j,
                          'OM': 1.54 + 0j,
                          'Soil': 1.56 + 0.01j,
                          'SS': 1.54 + 0j,
                          'BC': 1.80 + 0.54j,
                          'water': 1.333 + 0j}

        ext_dist, sca_dist, abs_dist = np.zeros((167,)), np.zeros((167,)), np.zeros((167,))
        ndp = np.array(ser[:np.size(self.dp)])

        for _m, _specie in zip(refractive_dic.values(),
                               ['AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio', 'Soil_volume_ratio',
                                'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']):
            _ndp = ser[_specie] / (1 + ser['ALWC_volume_ratio']) * ndp
            _Ext_dist, _Sca_dist, _Abs_dist = Mie_PESD(_m, 550, self.dp, self.dlogdp, _ndp)
            ext_dist += _Ext_dist
            sca_dist += _Sca_dist
            abs_dist += _Abs_dist

        return dict(ext=ext_dist, sca=sca_dist, abs=abs_dist)


if __name__ == '__main__':
    PNSD_data = SizeDist()
    PNSD_data.ext_process()
