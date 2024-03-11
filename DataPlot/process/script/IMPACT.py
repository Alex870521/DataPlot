from pandas import read_csv, concat
from ..core import *


class ImpactProcessor(DataProcessor):
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
    >>> df = ImpactProcessor(reset=True, filename='IMPACT.csv').process_data()

    """

    def __init__(self, reset=False, filename=None):
        super().__init__(reset)
        self.file_path = self.default_path / 'Level1' / filename

    def process_data(self):
        if self.file_path.exists() and not self.reset:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return read_csv(f, parse_dates=['Time']).set_index('Time')
        else:
            _df = concat([DataReader('Optical.csv'),
                          DataReader('PBLH.csv'),
                          DataReader('OCEC.csv'),
                          DataReader('Teom.csv'),
                          DataReader('ISORROPIA.csv')], axis=1)

            _df.to_csv(self.file_path)
            return _df
