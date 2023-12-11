from pathlib import Path
from pandas import read_csv, DataFrame
import pkg_resources
path_meta = Path(pkg_resources.resource_filename('DataPlot', 'Data-example'))


class DataProcessorBase:
    PATH_MAIN = Path(__file__).parents[2] / 'Data-example'

    def __init__(self, reset=False,
                 minion_path=None,
                 impact_path=None,
                 chemical_path=None):

        self.PATH_MAIN = self.PATH_MAIN
        self.reset = reset
        self.minion_path   = minion_path
        self.impact_path   = impact_path
        self.chemical_path = chemical_path
        self.minion = DataFrame()
        self.impact = DataFrame()
        self.chemical = DataFrame()
        self.improve = DataFrame()
        self.PSD = DataFrame()
        self.PESD = DataFrame()

    def load_data(self):
        if self.minion_path:
            with open(self.minion_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.minion = read_csv(f, parse_dates=['Time'], na_values=['-', 'E', 'F']).set_index('Time')

        if self.impact_path:
            with open(self.impact_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.impact = read_csv(f, parse_dates=['Time']).set_index('Time')

        if self.chemical_path:
            with open(self.chemical_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.chemical = read_csv(f, parse_dates=['Time']).set_index('Time')

    def process_data(self):
        # Implement data processing logic here
        pass

    def main(self):
        # Implement the main processing logic here
        pass

    def save_result(self, data):
        # Implement logic to save the result to a CSV file
        pass
