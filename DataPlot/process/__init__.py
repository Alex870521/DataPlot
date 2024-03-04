
from .core import DataReader, DataProcessor, timer, HourClassifier, StateClassifier, SeasonClassifier
from .method import Mie_PESD, other_process
from .script import ImpactProcessor, ImproveProcessor, SizeDist, ChemicalProcessor
import numpy as np
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
            impact = ImpactProcessor(reset=False, filename='IMPACT.csv').process_data()

            # 3. Mass_volume
            chemical = ChemicalProcessor(reset=False, filename='chemical.csv').process_data()

            # 4. improve
            improve = ImproveProcessor(reset=False, filename='revised_IMPROVE.csv', version='revised').process_data()

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
    state = StateClassifier
    season = SeasonClassifier
    hour = HourClassifier
    DataBase = DataBase

    method_map = {
        'State': state,
        'Season': season,
        'Hour': hour
    }

    def __new__(cls, df: DataFrame,
                by: Literal["State", "Season", "Hour"],
                statistic: Literal["Table", "Array"] = 'Array'):

        if f'{by}' not in df.columns:
            _df = cls.method_map[by](DataBase)
            df = concat([df, _df[f'{by}']], axis=1)

        group = df.groupby(f'{by}')

        if statistic == 'Array':
            return cls.statistic_array(group)
        else:
            return cls.statistic_table(group)

    @staticmethod
    def statistic_array(group):
        _avg, _std = {}, {}

        for name, subdf in group:
            _avg[name] = np.array(subdf.mean(numeric_only=True))
            _std[name] = np.array(subdf.std(numeric_only=True))

        return _avg, _std

    @staticmethod
    def statistic_table(group):
        return group.mean(numeric_only=True), group.mean(numeric_only=True)
