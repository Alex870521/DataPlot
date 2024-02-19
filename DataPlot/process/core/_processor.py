from DataPlot.process.core._DEFAULT_PATH import *


class DataProcessor:
    DEFAULT_PATH = DEFAULT_PATH

    def __init__(self, reset: bool = False):
        self.DEFAULT_PATH: Path = self.DEFAULT_PATH
        self.reset: bool = reset

    def process_data(self):
        # Implement data process logic here
        pass
