import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv

from DataPlot.process.core import *
from DataPlot.process.method import *


class SizeDist:
    """
    data : DataFrame
        The processed PSD data stored as a pandas DataFrame.
    index : DatetimeIndex
        The index of the DataFrame representing time.
    dp : ndarray
        The array of particle diameters from the PSD data.
    dlogdp : ndarray
        The array of logarithmic particle diameter bin widths.
    """

    def __init__(self, filename: Path | str = None, data: DataFrame = None):
        self.data: DataFrame = DataReader(filename) if data is None else data

        self._dp = np.array(self.data.columns, dtype=float)
        self._dlogdp = np.full_like(self._dp, 0.014)
        self._index = self.data.index.copy()
        self._state = 'dlogdp'

    @property
    def dp(self) -> np.ndarray:
        return self._dp

    @dp.setter
    def dp(self, new_dp: np.ndarray):
        self._dp = new_dp

    @property
    def dlogdp(self) -> np.ndarray:
        return self._dlogdp

    @dlogdp.setter
    def dlogdp(self, new_dlogdp: np.ndarray):
        self._dlogdp = new_dlogdp

    @property
    def index(self):
        return self._index

    @property
    def state(self):
        return self._state

    @staticmethod
    def distribution_prop(dist, dp, dlogdp, weighting):
        dist = np.array(dist)
        total = np.sum(dist)

        gmd, gsd = geometric(dp, dist, total)
        _mode = dp[mode(dist)]
        ultra, accum, coarse = contribution(dp, dist, total)

        pleonasm_mapping = {
            'Number': 'n',
            'Surface': 's',
            'Volume': 'v',
            'Ext_internal': 'ext_in',
            'Sca_internal': 'sca_in',
            'Abs_internal': 'abs_in',
            'Ext_external': 'ext_ex',
            'Sca_external': 'sca_ex',
            'Abs_external': 'abs_ex',
        }

        w = pleonasm_mapping[weighting]

        return {w: total, f'GMD{w}': gmd, f'GSD{w}': gsd, f'mode_{w}': _mode[0],
                f'ultra_{w}': ultra, f'accum_{w}': accum, f'coarse_{w}': coarse}


class ParticleSizeDist(DataProcessor):
    """
    A class for process particle size distribution (PSD) data.

    Parameters
    ----------
    filename : str, optional
        The name of the PSD data file.
        Defaults to 'PNSD_dNdlogdp.csv' in the default path.

    Attributes
    ----------
    file_path : Path
        The directory path where the PSD data file is located.


    Methods
    -------
    number()
        Calculate number distribution properties.

    surface(filename='PSSD_dSdlogdp.csv')
        Calculate surface distribution properties.

    volume(filename='PVSD_dVdlogdp.csv')
        Calculate volume distribution properties.

    psd_process(filename='PSD.csv')
        Process and save overall PSD properties.

    Raises
    ------
    ValueError
        If the PSD data is empty or missing.

    DataTypeError
        The input data unit must be dN/dlogdp.

    Examples
    --------
    Example 1: Use default path and filename
    >>> psd_data = ParticleSizeDist(filename='PNSD_dNdlogdp.csv').process_data(reset=True)
    """

    def __init__(self, filename: str = 'PNSD_dNdlogdp.csv'):
        super().__init__()
        self.file_path = self.DEFAULT_PATH / 'Level2'

        self._psd = SizeDist(filename)

    @property
    def number(self) -> SizeDist:
        """ Calculate number distribution """
        return self._psd

    @property
    def surface(self) -> SizeDist:
        """ Calculate surface distribution """
        return SizeDist(data=self._psd.data.dropna().apply(
            lambda col: math.pi * (self._psd.dp ** 2) * np.array(col), axis=1, result_type='broadcast'))

    @property
    def volume(self) -> SizeDist:
        """ Calculate volume distribution """
        return SizeDist(data=self._psd.data.dropna().apply(
            lambda col: math.pi / 6 * self._psd.dp ** 3 * np.array(col), axis=1, result_type='broadcast'))

    def process_data(self, reset: bool = False, save_filename: Path | str = 'PSD.csv'):
        file = self.file_path / save_filename
        if file.exists() and not reset:
            return read_csv(file, parse_dates=['Time']).set_index('Time')

        self.surface.data.reindex(self._psd.data.index).to_csv(self.file_path / 'PSSD_dSdlogdp.csv')
        self.volume.data.reindex(self._psd.data.index).to_csv(self.file_path / 'PVSD_dVdlogdp.csv')

        number_prop = self.number.data.dropna().apply(
            partial(SizeDist.distribution_prop, dp=self._psd.dp, dlogdp=self._psd.dlogdp, weighting='Number'), axis=1,
            result_type='expand')
        surface_prop = self.surface.data.dropna().apply(
            partial(SizeDist.distribution_prop, dp=self._psd.dp, dlogdp=self._psd.dlogdp, weighting='Surface'), axis=1,
            result_type='expand')
        volume_prop = self.volume.data.dropna().apply(
            partial(SizeDist.distribution_prop, dp=self._psd.dp, dlogdp=self._psd.dlogdp, weighting='Volume'), axis=1,
            result_type='expand')

        result_df = pd.concat([number_prop, surface_prop, volume_prop], axis=1).reindex(self._psd.index)
        result_df.to_csv(file)
        return result_df

    def save_data(self, data: DataFrame, save_filename: str | Path):
        data.to_csv(save_filename)


class ExtinctionDist(DataProcessor):

    def __init__(self, filename='PNSD_dNdlogdp.csv'):
        super().__init__()
        self.file_path = self.DEFAULT_PATH / 'Level2'

        self.psd = SizeDist(filename)
        self.RI = DataReader('chemical.csv')[['n_dry', 'n_amb', 'k_dry', 'k_amb',
                                              'AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio',
                                              'Soil_volume_ratio', 'SS_volume_ratio', 'EC_volume_ratio',
                                              'ALWC_volume_ratio']]

        self.data = pd.concat([self.psd, self.RI], axis=1).dropna()

    def extinction_internal(self, filename='PESD_dextdlogdp_internal.csv'):
        muti_df = self.data.apply(partial(internal, dp=self.dp, dlogdp=self.dlogdp), axis=1, result_type='expand')

        ext_dist = pd.DataFrame(muti_df['ext'].tolist(), index=muti_df['ext'].index).set_axis(self.dp, axis=1)
        sca_dist = pd.DataFrame(muti_df['sca'].tolist(), index=muti_df['sca'].index).set_axis(self.dp, axis=1)
        abs_dist = pd.DataFrame(muti_df['abs'].tolist(), index=muti_df['abs'].index).set_axis(self.dp, axis=1)

        ext_dist.reindex(self.index).to_csv(self.file_path / filename)

    def extinction_external(self, filename='PESD_dextdlogdp_external.csv'):
        muti_df = self.data.apply(external, args=(self.dp, self.dlogdp), axis=1, result_type='expand')

        ext_dist = pd.DataFrame(muti_df['ext'].tolist(), index=muti_df['ext'].index).set_axis(self.dp, axis=1)
        sca_dist = pd.DataFrame(muti_df['sca'].tolist(), index=muti_df['sca'].index).set_axis(self.dp, axis=1)
        abs_dist = pd.DataFrame(muti_df['abs'].tolist(), index=muti_df['abs'].index).set_axis(self.dp, axis=1)

        ext_dist.reindex(self.index).to_csv(self.file_path / filename)

    def extinction_sensitivity(self):
        Fixed_PNSD = np.array(data.iloc[:, :len(self.dp)].mean())
        Fixed_m = np.array(data['n_amb'].mean() + 1j * data['k_amb'].mean())

        fix_PNSD_ = data.apply(fix_PNSD, args=(self.dp, self.dlogdp, Fixed_PNSD), axis=1, result_type='expand')
        fix_RI_ = data.apply(fix_RI, args=(self.dp, self.dlogdp, Fixed_m), axis=1, result_type='expand')

        return pd.concat([fix_PNSD_, fix_RI_], axis=1).rename(columns={0: 'Bext_Fixed_PNSD', 1: 'Bext_Fixed_RI'})

    def process_data(self, reset: bool = False, save_filename: str | Path = 'PESD.csv'):
        file = self.file_path / save_filename
        if file.exists() and not reset:
            return read_csv(file, parse_dates=['Time']).set_index('Time')

        result_df = pd.concat([self.extinction_internal(), self.extinction_external(), self.extinction_sensitivity()],
                              axis=1).reindex(self.index)
        result_df.to_csv(file)
        return result_df

    def save_data(self, data: DataFrame, save_filename: str | Path):
        data.to_csv(save_filename)


if __name__ == '__main__':
    df = ParticleSizeDist(filename='PNSD_dNdlogdp.csv').process_data(reset=True)
