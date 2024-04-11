from abc import ABC, abstractmethod

from pandas import DataFrame

from DataPlot.process.core._DEFAULT_PATH import *


class DataProcessor(ABC):

    def __init__(self):
        self.DEFAULT_PATH: Path = DEFAULT_PATH

    @abstractmethod
    def process_data(self, reset: bool = False, save_filename: str | Path = None) -> DataFrame:
        """ Implementation of processing data """
        pass

    @abstractmethod
    def save_data(self, data: DataFrame, save_filename: str | Path):
        """ Save data """
        pass
