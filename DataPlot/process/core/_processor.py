from DataPlot.process.core._DEFAULT_PATH import *


class DataProcessor:

    def __init__(self, reset: bool = False, default_path: Path = DEFAULT_PATH):
        self.reset: bool = reset
        self.default_path: Path = default_path

    def process_data(self):
        # Implement data process logic here
        pass
