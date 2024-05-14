from abc import ABC, abstractmethod

from pandas import DataFrame

from DataPlot.process.core._DEFAULT import *


class DataProc(ABC):
    def __init__(self):
        self.DEFAULT_DATA_PATH: Path = DEFAULT_DATA_PATH

    @abstractmethod
    def process_data(self, reset: bool = False, save_filename: str | Path = None) -> DataFrame:
        """ Implementation of processing data """
        pass
