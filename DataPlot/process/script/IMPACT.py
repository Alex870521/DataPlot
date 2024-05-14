from pathlib import Path

from pandas import DataFrame, read_csv, concat

from DataPlot.process.core import DataReader, DataProc


class ImpactProc(DataProc):
    """
    A class for processing impact data.

    Parameters:
    -----------
    reset : bool, optional
        If True, resets the processing. Default is False.
    save_filename : str or Path, optional
        The name or path to save the processed data. Default is 'IMPACT.csv'.

    Methods:
    --------
    process_data(reset: bool = False, save_filename: str | Path = 'IMPACT.csv') -> DataFrame:
        Process data and save the result.

    save_data(data: DataFrame, save_filename: str | Path):
        Save processed data to a file.

    Attributes:
    -----------
    DEFAULT_PATH : Path
        The default path for data files.

    Examples:
    ---------
    >>> df_custom = ImpactProc().process_data(reset=True, save_filename='custom_file.csv')
    """

    def __init__(self):
        super().__init__()
        self.file_path = self.DEFAULT_DATA_PATH / 'Level1'

    def process_data(self, reset: bool = False, save_filename: str | Path = 'IMPACT.csv') -> DataFrame:
        file = self.file_path / save_filename
        if file.exists() and not reset:
            return read_csv(file, parse_dates=['Time']).set_index('Time')
        else:
            data_files = ['Optical.csv', 'PBLH.csv', 'OCEC.csv', 'Teom.csv', 'ISORROPIA.csv']
            _df = concat([DataReader(file) for file in data_files], axis=1)
            _df.to_csv(file)
            return _df


if __name__ == '__main__':
    df = ImpactProc().process_data(reset=True, save_filename='IMPACT.csv')
