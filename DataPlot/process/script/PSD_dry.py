import numpy as np
from pandas import read_csv, DataFrame

from DataPlot.process.script.PSD import ParticleSizeDist
from ..core import *


class DryPSDProcessor(DataProcessor):
    """
    A class for process impact data.

    Parameters:
    -----------
    reset : bool, optional
        If True, resets the process. Default is False.
    filename : str, optional
        The name of the file to process. Default is None.

    Methods:
    --------
    process_data():
        Process data and save the result.

    Attributes:
    -----------
    DEFAULT_PATH : Path
        The default path for data files.

    Examples:
    ---------
    >>> df = DryPSDProcessor(reset=True, filename='PNSD_dNdlogdp_dry.csv').process_data()

    """

    def __init__(self, reset=False, filename=None):
        super().__init__(reset)
        self.file_path = self.default_path / 'Level2' / 'distribution'

    def process_data(self):
        if self.file_path.exists() and not self.reset:
            return read_csv(self.file_path, parse_dates=['Time']).set_index('Time')

        else:
            PNSD = DataReader('PNSD_dNdlogdp.csv')
            chemical = DataReader('chemical.csv')

            # _df.to_csv(self.file_path)
            # return _df


psd = ParticleSizeDist(reset=True, filename='PNSD_dNdlogdp.csv')
breakpoint()


def dry_PNSD_process(**kwargs):
    index = df.index.copy()
    df_input = df.dropna()
    _index = df_input.index.copy()

    out_dis = {'dry_dist': [],
               }

    for _tm, _ser in df_input.iterrows():
        ndp = np.array(_ser[:_length])
        # uniform gRH
        gRH = resloved_gRH(_length, _ser['gRH'], uniform=True)

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

        dry_ndp = np.array(dry_ndp)
        new_dry_ndp = np.zeros(_length)
        new_dry_ndp[:dry_ndp.shape[0]] = dry_ndp

        out_dis['dry_dist'].append(new_dry_ndp)

    dry_PNSD = DataFrame(out_dis['dry_dist']).set_index(_index).set_axis(dp, axis=1).reindex(index)

    return dry_PNSD


def resloved_gRH(length, gRH=1.31, uniform=True):
    if uniform:
        arr = np.array([gRH] * length)

    else:
        lognorm_dist = lambda x, geoMean, geoStd: (gRH / (np.log10(geoStd) * np.sqrt(2 * np.pi))) * np.exp(-(x - np.log10(geoMean))**2 / (2 * np.log10(geoStd)**2))
        abc = lognorm_dist(np.log10(dp/1000), 0.5, 2.0)
        arr = np.where(abc < 1, 1, abc)

    return arr

