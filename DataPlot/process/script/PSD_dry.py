from pathlib import Path

import numpy as np
from pandas import DataFrame

from DataPlot.process.core import *
from DataPlot.process.script.SizeDist import SizeDist


class DryPSDProcessor(DataProcessor):
    """
    A class for process impact data.

    Parameters
    ----------
    reset : bool, optional
        If True, resets the process. Default is False.
    filename : str, optional
        The name of the file to process. Default is None.

    Methods
    -------
    process_data():
        Process data and save the result.

    Attributes
    ----------
    DEFAULT_PATH : Path
        The default path for data files.


    Examples
    --------
    >>> df = DryPSDProcessor(reset=True, filename='PNSD_dNdlogdp_dry.csv').process_data()
    """

    def __init__(self, filename: str = 'PNSD_dNdlogdp.csv'):
        super().__init__()

        self.file_path = self.DEFAULT_DATA_PATH / 'Level2'
        self.psd = SizeDist(filename)

    def process_data(self, reset: bool = False, save_filename: str | Path = None) -> DataFrame:
        # if self.file_path.exists() and not reset:
        #     return read_csv(self.file_path, parse_dates=['Time']).set_index('Time')

        PNSD = DataReader('PNSD_dNdlogdp.csv')
        chemical = DataReader('chemical.csv')

        # _df.to_csv(self.file_path)
        # return _df


def dry_PNSD_process(dist, dp, **kwargs):
    ndp = np.array(dist[:np.size(dp)])
    gRH = resolved_gRH(dp, dist['gRH'], uniform=True)

    dry_dp = dp / gRH
    belong_which_ibin = np.digitize(dry_dp, dp) - 1

    result = {}
    for i, (ibin, dn) in enumerate(zip(belong_which_ibin, ndp)):
        if dp[ibin] not in result:
            result[dp[ibin]] = []
        result[dp[ibin]].append(ndp[i])

    dry_ndp = []
    for key, val in result.items():
        dry_ndp.append(sum(val) / len(val))

    return np.array(dry_ndp)


def resolved_gRH(dp, gRH=1.31, uniform=True):
    if uniform:
        return np.array([gRH] * dp.size)

    else:
        lognorm_dist = lambda x, geoMean, geoStd: (gRH / (np.log10(geoStd) * np.sqrt(2 * np.pi))) * np.exp(-(x - np.log10(geoMean))**2 / (2 * np.log10(geoStd)**2))
        abc = lognorm_dist(np.log10(dp), 200, 2.0)
        return np.where(abc < 1, 1, abc)


if __name__ == '__main__':
    pass
