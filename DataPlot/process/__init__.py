import numpy as np
from pathlib import Path
from typing import Literal
from pandas import read_csv, concat, DataFrame
from tqdm import tqdm
from .core import DataReader, DataProcessor, Classifier
from .script import ImpactProcessor, ImproveProcessor, ChemicalProcessor, SizeDist, OthersProcessor

__all__ = ['DataBase',
           'DataReader',
           'DataClassifier',
           'SizeDist'
           ]


class MainProcessor(DataProcessor):
    def __init__(self, reset=False, filename='All_data.csv'):
        super().__init__(reset)
        self.file_path = Path(__file__).parents[1] / 'data' / filename

    def process_data(self):
        with tqdm(total=20, desc="Loading Data", unit="step") as progress_bar:

            if self.file_path.exists() and not self.reset:
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    progress_bar.update(20)
                    return read_csv(f, parse_dates=['Time'], low_memory=False).set_index('Time')
            else:
                # 1. EPB
                minion = DataReader('EPB.csv')
                progress_bar.update(1)

                # 2. IMPACT
                impact = ImpactProcessor(reset=False, filename='IMPACT.csv').process_data()
                progress_bar.update(1)

                # 3. Mass_volume
                chemical = ChemicalProcessor(reset=False, filename='chemical.csv').process_data()
                progress_bar.update(1)

                # 4. improve
                improve = ImproveProcessor(reset=False, filename='revised_IMPROVE.csv', version='revised').process_data()
                progress_bar.update(5)

                # 5. Number & Surface & volume & Extinction distribution
                psd = SizeDist(reset=False, filename='PNSD_dNdlogdp.csv').process_data()
                progress_bar.update(11)

                _df = concat([minion, impact, chemical, improve, psd], axis=1)

                # 6. others
                _df = OthersProcessor(reset=False, data=_df).process_data()
                progress_bar.update(1)

                # 7. save result
                _df.to_csv(self.file_path)

                return _df.copy()


DataBase = MainProcessor(reset=False).process_data()


class DataClassifier(Classifier):

    def __new__(cls, df: DataFrame,
                by: Literal["Hour", "State", "Season", "Season_state"],
                statistic: Literal["Table", "Array"] = 'Array'):

        if f'{by}' not in df.columns:
            _df = cls.classify(DataBase)
            df = concat([df, _df[f'{by}']], axis=1)

        group = df.groupby(f'{by}')

        if statistic == 'Array':
            return cls.statistic_array(group)
        else:
            return cls.statistic_table(group)

    @classmethod
    def classify(cls, df) -> DataFrame:
        df = df.copy()
        df['Month'] = df.index.strftime('%Y-%m')
        df['Hour'] = df.index.hour
        df['Diurnal'] = df['Hour'].apply(cls.map_diurnal)

        clean_upp_boud, event_low_boud = df.Extinction.quantile([0.2, 0.8])

        df['State'] = df.apply(cls.map_state, axis=1, clean_upp_boud=clean_upp_boud, event_low_boud=event_low_boud)

        for season, (season_start, season_end) in cls.Seasons.items():
            df.loc[season_start:season_end, 'Season'] = season

        for _grp, _df in df.groupby('Season'):
            clean_upp_boud, event_low_boud = _df.Extinction.quantile([0.2, 0.8])
            df['Season_State'] = df.apply(cls.map_state, axis=1, clean_upp_boud=clean_upp_boud,
                                          event_low_boud=event_low_boud)

        return df

    @staticmethod
    def map_diurnal(hour):
        if 7 <= hour <= 18:
            return 'Day'
        elif 19 <= hour <= 23:
            return 'Night'
        elif 0 <= hour <= 6:
            return 'Night'

    @staticmethod
    def map_state(row, clean_upp_boud, event_low_boud):
        if row['Extinction'] >= event_low_boud:
            return 'Event'
        elif clean_upp_boud < row['Extinction'] < event_low_boud:
            return 'Transition'
        else:
            return 'Clean'

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

    @staticmethod
    def returnStateDict(group):
        dic_grp_sea = {}
        for _grp, _df in group:
            cond_event = _df.State == 'Event'
            cond_transition = _df.State == 'Transition'
            cond_clean = _df.State == 'Clean'

            dic_grp_sea[_grp] = {'Total': _df.copy(),
                                 'Clean': _df.loc[cond_clean].copy(),
                                 'Transition': _df.loc[cond_transition].copy(),
                                 'Event': _df.loc[cond_event].copy()}

        return dic_grp_sea
