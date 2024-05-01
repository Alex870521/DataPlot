from abc import ABC, abstractmethod

from pandas import read_csv, read_json, read_excel, DataFrame

from DataPlot.process.core._DEFAULT import *


class FileHandler(ABC):
    """ An abstract base class for reading data files with different extensions (.csv, .json, .xls, .xlsx). """

    @abstractmethod
    def read_data(self, file_path: Path) -> DataFrame:
        pass

    @abstractmethod
    def save_data(self, data: DataFrame, file_path: Path) -> None:
        pass


class CsvFileHandler(FileHandler):
    def read_data(self, file_path: Path) -> DataFrame:
        return read_csv(file_path, na_values=('-', 'E', 'F'), parse_dates=['Time'], low_memory=False).set_index('Time')

    def save_data(self, data: DataFrame, file_path: Path) -> None:
        data.to_csv(file_path)


class JsonFileHandler(FileHandler):
    def read_data(self, file_path: Path) -> DataFrame:
        return read_json(file_path)

    def save_data(self, data: DataFrame, file_path: Path) -> None:
        # Implement JSON data saving logic here
        pass


class ExcelFileHandler(FileHandler):
    def read_data(self, file_path: Path) -> DataFrame:
        return read_excel(file_path, parse_dates=['Time'])

    def save_data(self, data: DataFrame, file_path: Path) -> None:
        # Implement JSON data saving logic here
        pass


class FileFinder:

    @staticmethod
    def find_file(filename: Path | str) -> Path:
        if isinstance(filename, str):
            file_path = list(DEFAULT_DATA_PATH.glob('**/' + filename))
            if len(file_path) == 1:
                return file_path[0]
            elif len(file_path) == 0:
                raise FileNotFoundError(f"File '{filename}' not found.")
            else:
                raise ValueError("Expected exactly one file, but found multiple files.")

        else:
            if not filename.exists():
                raise FileNotFoundError(f"File '{filename}' does not exist.")
            return filename


class DataReaderFactory:
    _handler_mapping = {
        '.csv': CsvFileHandler(),
        '.json': JsonFileHandler(),
        '.xls': ExcelFileHandler(),
        '.xlsx': ExcelFileHandler(),
    }

    @staticmethod
    def create_handler(file_extension: str) -> FileHandler:
        reader_class = DataReaderFactory._handler_mapping.get(file_extension)
        if reader_class is None:
            raise ValueError(f"Unsupported file format: {file_extension}")
        return reader_class


class DataReader:
    """
    A class for reading data files with different extensions (.csv, .json, .xls, .xlsx).

    Parameters
    ----------
        filename (Path | str): The name of the file to be read or the Path of the file.

    Returns
    -------
        pandas.DataFrame: data

    Examples
    --------
    >>> psd = DataReader('PNSD_dNdlogdp.csv')
    """

    def __new__(cls, filename: Path | str) -> DataFrame:
        file_path = FileFinder.find_file(filename)
        reader = DataReaderFactory.create_handler(file_path.suffix.lower())
        return reader.read_data(file_path)

    @staticmethod
    def read_data(filename: Path | str) -> DataFrame:
        file_path = FileFinder.find_file(filename)
        handler = DataReaderFactory.create_handler(file_path.suffix.lower())

        return handler.read_data(file_path)

    @staticmethod
    def save_data(data: DataFrame, filename: Path | str) -> None:
        file_path = filename
        handler = DataReaderFactory.create_handler(file_path.suffix.lower())

        handler.save_data(data, file_path)


if __name__ == '__main__':
    df = DataReader.read_data('EPB.csv')
