from pandas import DataFrame, concat, read_csv
from pathlib import Path
from core import DataReader, DataProcessor
from decorator import timer
from method import *
from functools import partial
import math
import pandas as pd


class SizeDist(DataProcessor):
    """
    A class for processing particle size distribution (PSD) data.

    Parameters
    ----------
    filename : str, optional
        The name of the PSD data file.
        Defaults to 'PNSD_dNdlogdp.csv' in the default path.

    Attributes
    ----------
    file_path : Path
        The directory path where the PSD data file is located.
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

    Raises
    ------
    ValueError
        If the PSD data is empty or missing.

    DataTypeError
        The input data unit must be dN/dlogdp.

    Examples
    --------
    Example 1: Use default path and filename
    >>> psd_data = SizeDist(reset=True, filename='PNSD_dNdlogdp.csv')
    >>> psd_data.psd_process()
    """

    def __init__(self, reset=False, filename=None):
        super().__init__(reset)
        self.file_path = super().DEFAULT_PATH / 'Level2' / 'distribution'

        self.data: pd.DataFrame = DataReader(filename).dropna()

        self.index = self.data.index.copy()
        self.dp = np.array(self.data.columns, dtype='float')
        self.dlogdp = np.full_like(self.dp, 0.014)

    def number(self, filename='PNSD_dSdlogdp.csv'):
        """
        Calculate number distribution.

        Returns
        -------
        result : ...
            Description of the result.

        """
        num_dist = self.data
        num_prop = num_dist.apply(partial(self.__dist_prop, weighting='Number'), axis=1, result_type='expand')

        return num_prop

    def surface(self, filename='PSSD_dSdlogdp.csv'):
        """
        Calculate surface distribution.

        Returns
        -------
        result : ...
            Description of the result.

        """
        surf_dist = self.data.apply(lambda col: math.pi * (self.dp ** 2) * np.array(col), axis=1, result_type='broadcast')
        surf_prop = surf_dist.apply(partial(self.__dist_prop, weighting='Surface'), axis=1, result_type='expand')

        surf_dist.reindex(self.index).to_csv(self.file_path / filename)

        return surf_prop

    def volume(self, filename='PVSD_dVdlogdp.csv'):
        """
        Calculate volume distribution.

        Returns
        -------
        result : ...
            Description of the result.

        """
        vol_dist = self.data.apply(lambda col: math.pi / 6 * self.dp ** 3 * np.array(col), axis=1, result_type='broadcast')
        vol_prop = vol_dist.apply(partial(self.__dist_prop, weighting='Volume'), axis=1, result_type='expand')

        vol_dist.reindex(self.index).to_csv(self.file_path / filename)

        return vol_prop

    def extinction_internal(self, filename='PESD_dextdlogdp_internal.csv'):
        psd_data, m_data = self.data, DataReader('chemical.csv')[['n_amb', 'k_amb']]
        data = pd.concat([psd_data, m_data], axis=1).dropna()

        muti_df = data.apply(partial(internal, dp=self.dp, dlogdp=self.dlogdp), axis=1, result_type='expand')

        ext_dist = pd.DataFrame(muti_df['ext'].tolist(), index=muti_df['ext'].index).set_axis(self.dp, axis=1)
        sca_dist = pd.DataFrame(muti_df['sca'].tolist(), index=muti_df['sca'].index).set_axis(self.dp, axis=1)
        abs_dist = pd.DataFrame(muti_df['abs'].tolist(), index=muti_df['abs'].index).set_axis(self.dp, axis=1)

        ext_prop = ext_dist.apply(partial(self.__dist_prop, weighting='Ext_internal'), axis=1, result_type='expand')
        sca_prop = sca_dist.apply(partial(self.__dist_prop, weighting='Sca_internal'), axis=1, result_type='expand')
        abs_prop = abs_dist.apply(partial(self.__dist_prop, weighting='Abs_internal'), axis=1, result_type='expand')

        ext_dist.reindex(self.index).to_csv(self.file_path / filename)

        return pd.concat([ext_prop, sca_prop['Bsca_internal'], abs_prop['Babs_internal']], axis=1)

    def extinction_external(self, filename='PESD_dextdlogdp_external.csv'):
        psd_data, m_data = self.data, DataReader('chemical.csv')[
            ['AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio', 'Soil_volume_ratio',
             'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']]
        data = pd.concat([psd_data, m_data], axis=1).dropna()

        muti_df = data.apply(external, args=(self.dp, self.dlogdp), axis=1, result_type='expand')

        ext_dist = pd.DataFrame(muti_df['ext'].tolist(), index=muti_df['ext'].index).set_axis(self.dp, axis=1)
        sca_dist = pd.DataFrame(muti_df['sca'].tolist(), index=muti_df['sca'].index).set_axis(self.dp, axis=1)
        abs_dist = pd.DataFrame(muti_df['abs'].tolist(), index=muti_df['abs'].index).set_axis(self.dp, axis=1)

        ext_prop = ext_dist.apply(partial(self.__dist_prop, weighting='Ext_external'), axis=1, result_type='expand')
        sca_prop = sca_dist.apply(partial(self.__dist_prop, weighting='Sca_external'), axis=1, result_type='expand')
        abs_prop = abs_dist.apply(partial(self.__dist_prop, weighting='Abs_external'), axis=1, result_type='expand')

        ext_dist.reindex(self.index).to_csv(self.file_path / filename)

        return pd.concat([ext_prop, sca_prop['Bsca_external'], abs_prop['Babs_external']], axis=1)

    def process_data(self):
        pass

    @timer
    def psd_process(self, reset=None, filename='PSD.csv'):
        file = self.file_path.parent / filename
        if file.exists() and not self.reset:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                return read_csv(f, parse_dates=['Time']).set_index('Time')

        result_df = pd.concat([self.number(), self.surface(), self.volume()], axis=1).reindex(self.index)
        result_df.to_csv(self.file_path.parent / filename)
        return result_df

    @timer
    def ext_process(self, reset=None, filename='PESD.csv'):
        file = self.file_path.parent / filename
        if file.exists() and not self.reset:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                return read_csv(f, parse_dates=['Time']).set_index('Time')

        result_df = pd.concat([self.extinction_internal(), self.extinction_external()], axis=1).reindex(self.index)
        result_df.to_csv(self.file_path.parent / filename)
        return result_df

    def __dist_prop(self, ser, weighting):
        dist = np.array(ser) * self.dlogdp
        total = np.sum(dist)

        gmd, gsd = geometric(self.dp, dist, total)
        _mode = mode(self.dp, dist)
        ultra, accum, coarse = contribution(self.dp, dist, total)

        weight_mapping = {
            'Number': 'Number',
            'Surface': 'Surface',
            'Volume': 'Volume',
            'Ext_internal': 'Bext_internal',
            'Sca_internal': 'Bsca_internal',
            'Abs_internal': 'Babs_internal',
            'Ext_external': 'Bext_external',
            'Sca_external': 'Bsca_external',
            'Abs_external': 'Babs_external',
        }

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

        return {weight_mapping[weighting]: total, f'GMD{w}': gmd, f'GSD{w}': gsd,
                f'mode_{w}': _mode, f'ultra_{w}': ultra, f'accum_{w}': accum, f'coarse_{w}': coarse}


if __name__ == '__main__':
    psd = SizeDist(reset=True, filename='PNSD_dNdlogdp.csv')
    psd.ext_process()
    # internal + external = 416.99s
    # internal = 56.19s -> 呼叫Mie越多次越慢