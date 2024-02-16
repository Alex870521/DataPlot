from pathlib import Path


class DataProcessor:
    DEFAULT_PATH = Path(__file__).parents[2] / 'Data-example'

    def __init__(self, reset: bool = False):
        self.DEFAULT_PATH = self.DEFAULT_PATH
        self.reset = reset

    def process_data(self):
        # Implement data process logic here
        pass
