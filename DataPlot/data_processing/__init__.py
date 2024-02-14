
# core object
from .core import *

# method object
from .method import *

# process class
from .script import *

from pathlib import Path
from pandas import read_csv, concat


class MainProcessor(DataProcessor):
    def __init__(self, reset=False, filename='All_data.csv'):
        super().__init__(reset)
        self.file_path = Path(__file__).parents[1] / 'Data-example' / filename

    @timer
    def process_data(self):
        if self.file_path.exists() and not self.reset:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return read_csv(f, parse_dates=['Time'], low_memory=False).set_index('Time')
        else:
            # 1. EPB
            minion = DataReader('EPB.csv')

            # 2. IMPACT
            impact = ImpactProcessor(reset=self.reset, filename='IMPACT.csv').process_data()

            # 3. Mass_volume
            chemical = ChemicalProcessor(reset=self.reset, filename='chemical.csv').process_data()

            # 4. IMPROVE
            improve = ImproveProcessor(reset=self.reset, filename='revised_IMPROVE.csv', version='revised').process_data()

            # 5. Number & Surface & volume & Extinction distribution
            PSD = SizeDist(reset=False, filename='PNSD_dNdlogdp.csv')

            psd = PSD.psd_process()
            ext = PSD.ext_process()

            # Extinction_PNSD_dry = dataproc.Extinction_dry_PSD_internal_process(reset=False)
            _df = concat([minion, impact, chemical, improve, psd, ext], axis=1)

            # 7. others
            _df = other_process(_df.copy())

            # 8. save result
            _df.to_csv(self.file_path)

            return _df


data = MainProcessor(reset=True).process_data()
