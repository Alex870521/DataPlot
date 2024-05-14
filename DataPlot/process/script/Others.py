import numpy as np
import pandas as pd

from DataPlot.process.core import DataProc


class OthersProc(DataProc):
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
    >>> df = OthersProc(reset=True, filename=None).process_data()

    """

    def __init__(self, reset=False, data=None):
        super().__init__(reset)
        self.file_path = None
        self.data = data

    def process_data(self) -> pd.DataFrame:
        df = self.data
        df['PG'] = df[['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas']].dropna().copy().apply(np.sum, axis=1)
        df['MAC'] = df['Absorption'] / df['T_EC']
        df['Ox'] = df['NO2'] + df['O3']
        df['N2O5_tracer'] = df['NO2'] * df['O3']
        df['Vis_cal'] = 1096 / df['Extinction']
        # df['fRH_Mix'] = df['Bext'] / df['Extinction']
        # df['fRH_PNSD'] = df['Bext_internal'] / df['Bext_dry']
        df['fRH_IMPR'] = df['total_ext'] / df['total_ext_dry']
        df['OCEC_ratio'] = df['O_OC'] / df['O_EC']
        df['PM1/PM25'] = np.where(df['PM1'] / df['PM25'] < 1, df['PM1'] / df['PM25'], np.nan)
        df['MEE_PNSD'] = df['Bext_internal'] / df['PM25']
        # df['MEE_dry_PNSD'] = df['Bext_dry'] / df['PM25']
        return df
