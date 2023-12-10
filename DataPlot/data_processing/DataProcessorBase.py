from pathlib import Path
from pandas import read_csv
import pkg_resources
path_meta = Path(pkg_resources.resource_filename('DataPlot', 'Data-example'))


class DataProcessorBase:
    PATH_MAIN = Path(__file__).parents[2] / 'Data-example'

    def __init__(self, reset=False):
        self.PATH_MAIN = self.PATH_MAIN
        self.reset = reset
        self.minion = None
        self.impact = None
        self.chemical = None
        self.improve = None
        self.PSD = None
        self.PESD = None

    def process_data(self):
        # Implement data processing logic here
        pass

    def main(self):
        # Implement the main processing logic here
        pass

    def save_result(self, data):
        # Implement logic to save the result to a CSV file
        pass
