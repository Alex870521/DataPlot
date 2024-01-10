from pandas import DataFrame, concat
from pathlib import Path
from core import DataReader, DataProcessor
from decorator import timer
from method import *
from functools import partial


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

        Examples
        --------
        Example usage of the number method:

        >>> psd = SizeDist()
        >>> result = psd.number()
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

        Examples
        --------
        Example usage of the number method:

        >>> psd = SizeDist()
        >>> result = psd.surface()
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

        Examples
        --------
        Example usage of the number method:

        >>> psd = SizeDist()
        >>> result = psd.volume()
        """
        vol_dist = self.data.apply(lambda col: math.pi / 6 * self.dp ** 3 * np.array(col), axis=1, result_type='broadcast')
        vol_prop = vol_dist.apply(partial(self.__dist_prop, weighting='Volume'), axis=1, result_type='expand')

        vol_dist.reindex(self.index).to_csv(self.file_path / filename)

        return vol_prop

    def extinction_internal(self, filename='PESD_dextdlogdp_internal.csv'):
        ext_data = pd.concat([self.data, DataReader('chemical.csv')], axis=1).dropna(subset=['n_amb', 'k_amb'])

        result_dic = ext_data.apply(internal_ext_dist, args=(self.dp, self.dlogdp), axis=1, result_type='expand')

        ext_dist = pd.DataFrame(result_dic['ext'].tolist(), index=result_dic['ext'].index).set_axis(self.dp, axis=1)
        sca_dist = pd.DataFrame(result_dic['sca'].tolist(), index=result_dic['sca'].index).set_axis(self.dp, axis=1)
        abs_dist = pd.DataFrame(result_dic['abs'].tolist(), index=result_dic['abs'].index).set_axis(self.dp, axis=1)

        ext_prop = ext_dist.apply(self.__dist_prop, axis=1, result_type='expand')

        ext_dist.reindex(self.index).to_csv(self.file_path / filename)

        return pd.DataFrame({'Bext_internal': ext_dist.apply(np.sum, axis=1) * 0.014,
                             'Bsca_internal': sca_dist.apply(np.sum, axis=1) * 0.014,
                             'Babs_internal': abs_dist.apply(np.sum, axis=1) * 0.014,
                             'GMD_ext_in': ext_prop['GMD'],
                             'GSD_ext_in': ext_prop['GSD'],
                             'mode_ext_in': ext_prop['mode'], })

    def extinction_external(self, filename='PESD_dextdlogdp_external.csv'):
        fil_col = ['AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio', 'Soil_volume_ratio',
                   'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']
        ext_data = pd.concat([self.data, DataReader('chemical.csv')[
            ['AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio', 'Soil_volume_ratio',
             'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']]], axis=1).dropna()

        result_dic2 = ext_data.apply(external_ext_dist, args=(self.dp, self.dlogdp), axis=1, result_type='expand')

        ext_dist2 = pd.DataFrame(result_dic2['ext'].tolist(), index=result_dic2['ext'].index).set_axis(self.dp, axis=1)
        sca_dist2 = pd.DataFrame(result_dic2['sca'].tolist(), index=result_dic2['sca'].index).set_axis(self.dp, axis=1)
        abs_dist2 = pd.DataFrame(result_dic2['abs'].tolist(), index=result_dic2['abs'].index).set_axis(self.dp, axis=1)

        ext_prop2 = ext_dist2.apply(self.__dist_prop, axis=1, result_type='expand')

        ext_dist2.reindex(self.index).to_csv(self.file_path / filename)

        return pd.DataFrame({'Bext_external': ext_dist2.apply(np.sum, axis=1) * 0.014,
                             'Bsca_external': sca_dist2.apply(np.sum, axis=1) * 0.014,
                             'Babs_external': abs_dist2.apply(np.sum, axis=1) * 0.014,
                             'GMD_ext_ex': ext_prop2['GMD'],
                             'GSD_ext_ex': ext_prop2['GSD'],
                             'mode_ext_ex': ext_prop2['mode'], })

    @timer
    def psd_process(self, reset=None, filename='PSD.csv'):
        result_df = pd.concat([self.number(), self.surface(), self.volume()], axis=1).reindex(self.index)
        result_df.to_csv(self.file_path.parent / filename)
        return result_df

    @timer
    def ext_process(self, reset=None, filename='PESD.csv'):
        result_df = pd.concat([self.extinction_internal(), self.extinction_external(), ], axis=1).reindex(self.index)
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
            'Volume': 'Volume'
        }

        pleonasm_mapping = {
            'Number': 'n',
            'Surface': 's',
            'Volume': 'v'
        }

        w = pleonasm_mapping[weighting]

        return {weight_mapping[weighting]: total, f'GMD{w}': gmd, f'GSD{w}': gsd,
                f'mode_{w}': _mode, f'ultra_{w}': ultra, f'accum_{w}': accum, f'coarse_{w}': coarse}


if __name__ == '__main__':
    psd = SizeDist(reset=True, filename='PNSD_dNdlogdp.csv')
    psd.psd_process()