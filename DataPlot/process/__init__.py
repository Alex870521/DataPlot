
from .core import DataReader, DataProcessor, timer, HourClassifier, StateClassifier, SeasonClassifier
from .method import Mie_PESD, other_process
from .script import ImpactProcessor, ImproveProcessor, SizeDist, ChemicalProcessor
from numpy import array
from pathlib import Path
from pandas import read_csv, concat
from datetime import datetime
from pandas import DataFrame
from typing import Literal, Dict, Any

__all__ = ['DataBase',
           'DataReader',
           'SizeDist',
           'Seasons',
           'Classifier',
           'Mie_PESD'
           ]


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

            # 4. improve
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


DataBase = MainProcessor(reset=False).process_data()

Seasons = {'2020-Summer': (datetime(2020, 9, 4), datetime(2020, 9, 21, 23)),
           '2020-Autumn': (datetime(2020, 9, 22), datetime(2020, 12, 29, 23)),
           '2020-Winter': (datetime(2020, 12, 30), datetime(2021, 3, 25, 23)),
           '2021-Spring': (datetime(2021, 3, 26), datetime(2021, 5, 6, 23))}
           # '2021-Summer': (datetime(2021, 5, 7), datetime(2021, 10, 16, 23))
           # '2021-Autumn': (datetime(2021, 10, 17), datetime(2021, 12, 31, 23))


class Classifier:
    by = None
    state = StateClassifier
    season = SeasonClassifier
    hour = HourClassifier

    def __new__(cls, df: DataFrame, by: Literal["State", "Season", "Hour"]):
        cls.by = by
        if f'{by}' not in df.columns:
            if by == 'State':
                group = cls.state(DataBase)
            elif by == 'Season':
                group = cls.season(DataBase)
            else:
                group = cls.hour(DataBase)

            for _grp, _df in group:
                data = df.merge(_df[f'{by}'], left_index=True, right_index=True)



            breakpoint()
            group = df.groupby(f'{by}')

        else:
            if by == 'State':
                group = cls.state(df)
            elif by == 'Season':
                group = cls.season(df)
            else:
                group = cls.hour(df)

        return group.mean(numeric_only=True), group.mean(numeric_only=True)

    @classmethod
    def statistic(cls, instance, state='Total'):
        mean = instance[f'{state}'].groupby(f'{cls.by}').mean(numeric_only=True)
        std = instance[f'{state}'].groupby(f'{cls.by}').std(numeric_only=True)
        return mean, std

    def mark_status(self):
        pass
