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

    def __init__(self, filename: str):
        self.file_path: Union[Path, None] = list(self.DEFAULT_PATH.glob('**/' + filename))[0]
        self.data:      Union[DataFrame, None] = self.read_data(self.file_path)

    def __new__(cls, filename: str) -> 'DataFrame':
        file_path: Union[Path, None] = cls.find_file(filename)
        data:      Union[DataFrame, None] = cls.read_data(file_path)
        return data

    @classmethod
    def find_file(cls, filename: str) -> Path:
        try:
            file_path = list(cls.DEFAULT_PATH.glob('**/' + filename))[0]
        except IndexError:
            raise FileNotFound(f"File '{filename}' not found.")
        else:
            return file_path

    @classmethod
    def read_data(cls, file_path) -> Union[DataFrame, None]:
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
    def read_csv(file_path) -> DataFrame:
        return read_csv(file_path, parse_dates=['Time'], na_values=['-', 'E', 'F'], low_memory=False).set_index('Time')

    @staticmethod
    def read_json(file_path) -> DataFrame:
        return read_json(file_path)

    @staticmethod
    def read_excel(file_path) -> DataFrame:
        return read_excel(file_path, parse_dates=['Time'])
