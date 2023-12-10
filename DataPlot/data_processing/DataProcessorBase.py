from pathlib import Path
from pandas import read_csv
import pkg_resources
path_meta = Path(pkg_resources.resource_filename('DataPlot', 'Data-example'))


class DataProcessorBase:
    PATH_MAIN = path_meta

    with open(PATH_MAIN / 'level1' / 'EPB.csv', 'r', encoding='utf-8', errors='ignore') as f:
        minion = read_csv(f, parse_dates=['Time'], na_values=['-', 'E', 'F']).set_index('Time')

    with open(PATH_MAIN / 'level1' / 'IMPACT.csv', 'r', encoding='utf-8', errors='ignore') as f:
        impact = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'level2' / 'chemical.csv', 'r', encoding='utf-8', errors='ignore') as f:
        chemical = read_csv(f, parse_dates=['Time']).set_index('Time')

    def __init__(self, reset=False):
        self.PATH_MAIN = self.PATH_MAIN
        self.reset = reset
        self.minion = self.minion
        self.impact = self.impact
        self.chemical = self.chemical
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