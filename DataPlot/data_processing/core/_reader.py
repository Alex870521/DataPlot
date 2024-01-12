from typing import Union
from pathlib import Path
from pandas import read_csv, read_json, read_excel, DataFrame


class FileNotFound(Exception):
    pass


class DataReader:
    """
    A class for reading data files with different extensions (.csv, .json, .xls, .xlsx).

    Parameters
    ----------
        filename (str): The name of the file to be read.

    Attributes
    ----------
        file_path (Path): The full path to the file.

    Returns
    -------
        pandas.DataFrame: data

    Examples
    --------
    >>> psd = DataReader('PNSD_dNdlogdp.csv')

    >>> chemical = DataReader('chemical.csv')
    """

    DEFAULT_PATH = Path(__file__).parents[2] / 'Data-example'

    def __new__(cls, filename: str) -> Union[DataFrame, None]:
        try:
            file_path = list(cls.DEFAULT_PATH.glob('**/' + filename))[0]
        except IndexError:
            raise FileNotFound(f"File '{filename}' not found.")
        else:
            return cls.read_data(file_path)

    def __init__(self, filename):
        self.file_path = list(self.DEFAULT_PATH.glob('**/' + filename))[0]
        self.data: DataFrame = self.read_data(self.file_path)

    @classmethod
    def read_data(cls, file_path) -> DataFrame:
        file_extension = file_path.suffix.lower()

        if file_extension == '.csv':
            return cls.read_csv(file_path)
        elif file_extension == '.json':
            return cls.read_json(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            return cls.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    @staticmethod
    def read_csv(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time'], na_values=['-', 'E', 'F'], low_memory=False).set_index('Time')

    @staticmethod
    def read_json(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return read_json(f)

    @staticmethod
    def read_excel(file_path):
        return read_excel(file_path, parse_dates=['Time'])
