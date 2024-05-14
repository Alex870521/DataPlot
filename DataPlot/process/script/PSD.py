from pathlib import Path

from pandas import concat, read_csv

from DataPlot.process.core import DataReader, DataProc
from DataPlot.process.script.AbstractDistCalc import DistributionCalculator
from DataPlot.process.script.SizeDist import SizeDist


class ParticleSizeDistProc(DataProc):
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
    >>> psd_data = ParticleSizeDistProc(filename='PNSD_dNdlogdp.csv').process_data(reset=True)
    """

    def __init__(self, filename: str = 'PNSD_dNdlogdp.csv'):
        super().__init__()
        self.file_path = self.DEFAULT_DATA_PATH / 'Level2'

        self.psd = SizeDist(filename)

    def process_data(self, reset: bool = False, save_filename: Path | str = 'PSD.csv'):
        file = self.file_path / save_filename
        if file.exists() and not reset:
            return read_csv(file, parse_dates=['Time']).set_index('Time')

        number = DistributionCalculator('number', self.psd).useApply()
        surface = DistributionCalculator('surface', self.psd).useApply()
        volume = DistributionCalculator('volume', self.psd).useApply()

        surface.to_csv(self.file_path / 'PSSD_dSdlogdp.csv')
        volume.to_csv(self.file_path / 'PVSD_dVdlogdp.csv')

        result_df = concat(
            [DistributionCalculator('property', SizeDist(data=number, weighting='n')).useApply(),
             DistributionCalculator('property', SizeDist(data=surface, weighting='s')).useApply(),
             DistributionCalculator('property', SizeDist(data=volume, weighting='v')).useApply()
             ], axis=1)

        result_df.to_csv(self.file_path / save_filename)
        return result_df


class ExtinctionDistProc(DataProc):

    def __init__(self, filename: str = 'PNSD_dNdlogdp.csv'):
        super().__init__()
        self.file_path = self.DEFAULT_DATA_PATH / 'Level2'

        self.psd = SizeDist(filename)
        self.RI = DataReader('chemical.csv')[['n_dry', 'n_amb', 'k_dry', 'k_amb',
                                              'AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio',
                                              'Soil_volume_ratio', 'SS_volume_ratio', 'EC_volume_ratio',
                                              'ALWC_volume_ratio']]

    def process_data(self, reset: bool = False, save_filename: str | Path = 'PESD.csv'):
        file = self.file_path / save_filename
        if file.exists() and not reset:
            return read_csv(file, parse_dates=['Time']).set_index('Time')

        ext_internal = DistributionCalculator('extinction', self.psd, self.RI, method='internal',
                                              result_type='extinction').useApply()
        ext_external = DistributionCalculator('extinction', self.psd, self.RI, method='external',
                                              result_type='extinction').useApply()

        ext_internal.to_csv(self.file_path / 'PESD_dextdlogdp_internal.csv')
        ext_external.to_csv(self.file_path / 'PESD_dextdlogdp_external.csv')

        result_df = concat([
            DistributionCalculator('property', SizeDist(data=ext_internal, weighting='ext_in')).useApply(),
            DistributionCalculator('property', SizeDist(data=ext_internal, weighting='ext_ex')).useApply(),
        ], axis=1)

        result_df.to_csv(self.file_path / save_filename)
        return result_df


if __name__ == '__main__':
    df = ParticleSizeDistProc(filename='PNSD_dNdlogdp.csv').process_data(reset=False)
    # df = ExtinctionDistProcessor(filename='PNSD_dNdlogdp.csv').process_data(reset=True)
