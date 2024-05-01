from pathlib import Path

import pandas as pd

from DataPlot.process.core import *
from DataPlot.process.script.SizeDist import SizeDist


class ParticleSizeDistProcessor(DataProcessor):
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

    psd : SizeDist
        The SizeDist object.

    Methods
    -------
    process_data(filename='PSD.csv')
        Process and save overall PSD properties.

    Examples
    --------
    Example 1: Use default path and filename
    >>> psd_data = ParticleSizeDistProcessor(filename='PNSD_dNdlogdp.csv').process_data(reset=True)
    """

    def __init__(self, filename: str = 'PNSD_dNdlogdp.csv'):
        super().__init__()
        self.file_path = self.DEFAULT_DATA_PATH / 'Level2'

        self.psd = SizeDist(filename)

    def process_data(self, reset: bool = False, save_filename: Path | str = 'PSD.csv'):
        # file = self.file_path / save_filename
        # if file.exists() and not reset:
        #     return read_csv(file, parse_dates=['Time']).set_index('Time')

        result_df = pd.concat([
            SizeDist(data=self.psd.number(), weighting='n').properties(),
            SizeDist(data=self.psd.surface(save_filename=self.file_path / 'PSSD_dSdlogdp.csv'),
                     weighting='s').properties(),
            SizeDist(data=self.psd.volume(save_filename=self.file_path / 'PVSD_dVdlogdp.csv'),
                     weighting='v').properties()
        ], axis=1)

        result_df.to_csv(self.file_path / save_filename)
        return result_df


class ExtinctionDistProcessor(DataProcessor):

    def __init__(self, filename: str = 'PNSD_dNdlogdp.csv'):
        super().__init__()
        self.file_path = self.DEFAULT_DATA_PATH / 'Level2'

        self.psd = SizeDist(filename)
        self.RI = DataReader('chemical.csv')[['n_dry', 'n_amb', 'k_dry', 'k_amb',
                                              'AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio',
                                              'Soil_volume_ratio', 'SS_volume_ratio', 'EC_volume_ratio',
                                              'ALWC_volume_ratio']]

    def process_data(self, reset: bool = False, save_filename: str | Path = 'PESD.csv'):
        # file = self.file_path / save_filename
        # if file.exists() and not reset:
        #     return read_csv(file, parse_dates=['Time']).set_index('Time')

        result_df = pd.concat([
            SizeDist(data=self.psd.extinction(self.RI, method='internal', result_type='extinction',
                                              save_filename=self.file_path / 'PESD_dextdlogdp_internal.csv'),
                     weighting='ext_in').properties(),
            SizeDist(data=self.psd.extinction(self.RI, method='external', result_type='extinction',
                                              save_filename=self.file_path / 'PESD_dextdlogdp_external.csv'),
                     weighting='ext_ex').properties(),
        ], axis=1)

        result_df.to_csv(self.file_path / save_filename)
        return result_df


if __name__ == '__main__':
    df = ParticleSizeDistProcessor(filename='PNSD_dNdlogdp.csv').process_data(reset=True)
    # df = ExtinctionDistProcessor(filename='PNSD_dNdlogdp.csv').process_data(reset=True)
