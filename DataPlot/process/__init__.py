import numpy as np
from pathlib import Path
from typing import Literal
from pandas import DataFrame
from pandas import read_csv, concat

from .core import DataReader, DataProcessor, timer, Classifier, SEASONS
from .method import other_process
from .script import ImpactProcessor, ImproveProcessor, ChemicalProcessor, SizeDist

__all__ = ['DataBase',
           'DataReader',
           'SizeDist',
           'SEASONS',
           'Classify',
           ]


class MainProcessor(DataProcessor):
    def __init__(self, reset=False, filename='All_data.csv'):
        super().__init__(reset)
        self.file_path = Path(__file__).parents[1] / 'data' / filename

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

            return _df.copy()


DataBase = MainProcessor(reset=False).process_data()


class Classify(Classifier):
    Seasons = SEASONS
    DataBase = DataBase

    def __new__(cls, df: DataFrame,
                by: Literal["State", "Season", "Hour"],
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

        # dic_grp_sea = {}
        df_group_season = df.groupby('Season')

        for _grp, _df in df_group_season:
            clean_upp_boud, event_low_boud = _df.Extinction.quantile([0.2, 0.8])
            df['Season_State'] = df.apply(cls.map_state, axis=1, clean_upp_boud=clean_upp_boud, event_low_boud=event_low_boud)

        #     cond_event = _df.State == 'Event'
        #     cond_transition = _df.State == 'Transition'
        #     cond_clean = _df.State == 'Clean'
        #
        #     dic_grp_sea[_grp] = {'Total': _df.copy(),
        #                          'Clean': _df.loc[cond_clean].copy(),
        #                          'Transition': _df.loc[cond_transition].copy(),
        #                          'Event': _df.loc[cond_event].copy()}

        return df

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

