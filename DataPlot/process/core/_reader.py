from typing import Optional
from pandas import read_csv, read_json, read_excel, DataFrame
from DataPlot.process.core._DEFAULT_PATH import *


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

    DEFAULT_PATH = DEFAULT_PATH

    def __new__(cls, filename: str) -> DataFrame:
        file_path: Optional[Path] = cls.find_file(filename)
        data:      Optional[DataFrame] = cls.read_data(file_path)
        return data

    @classmethod
    def find_file(cls, filename: str) -> Path:
        try:
            file_path = list(cls.DEFAULT_PATH.glob('**/' + filename))[0]
            return file_path

        except IndexError:
            raise FileNotFound(f"File '{filename}' not found.")

    @classmethod
    def read_data(cls, file_path) -> Optional[DataFrame]:
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
    def read_csv(file_path: Path) -> DataFrame:
        return read_csv(file_path, na_values=('-', 'E', 'F'), parse_dates=['Time'], low_memory=False).set_index('Time')

    @staticmethod
    def read_json(file_path: Path) -> DataFrame:
        return read_json(file_path)

    @staticmethod
    def read_excel(file_path: Path) -> DataFrame:
        return read_excel(file_path, parse_dates=['Time'])


if __name__ == '__main__':
    df = DataReader('EPB.csv')
