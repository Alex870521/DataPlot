
from .core import DataReader, DataProcessor, timer
from .method import Mie_PESD, other_process
from .script import ImpactProcessor, ImproveProcessor, SizeDist, ChemicalProcessor

from pathlib import Path
from pandas import read_csv, concat
from datetime import datetime
from pandas import DataFrame


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


class _Classifier:
    def __new__(cls, df):
        return cls.classify(df)

    @classmethod
    def classify(cls, df):
        pass


class HourClassifier(_Classifier):
    def __new__(cls, df) -> DataFrame:
        return cls.classify(df)

    @classmethod
    def classify(cls, df) -> DataFrame:
        df['Month'] = df.index.strftime('%Y-%m')
        df['Hour'] = df.index.hour

        df.loc[(df['Hour'] <= 18) & (df['Hour'] >= 7), 'Diurnal'] = 'Day'
        df.loc[(df['Hour'] <= 23) & (df['Hour'] >= 19), 'Diurnal'] = 'Night'
        df.loc[(df['Hour'] <= 6) & (df['Hour'] >= 0), 'Diurnal'] = 'Night'

        return df


class SeasonClassifier(_Classifier):
    Seasons = Seasons

    def __new__(cls, df) -> dict:
        return cls.classify(df)

    @classmethod
    def classify(cls, df) -> dict:
        df['Month'] = df.index.strftime('%Y-%m')
        df['Hour'] = df.index.hour
        df.loc[(df['Hour'] <= 18) & (df['Hour'] >= 7), 'Diurnal'] = 'Day'
        df.loc[(df['Hour'] <= 23) & (df['Hour'] >= 19), 'Diurnal'] = 'Night'
        df.loc[(df['Hour'] <= 6) & (df['Hour'] >= 0), 'Diurnal'] = 'Night'

        for season, (season_start, season_end) in cls.Seasons.items():
            df.loc[season_start:season_end, 'Season'] = season

        dic_grp_sea = {}
        df_group_season = df.groupby('Season')

        for _grp, _df in df_group_season:
            clean_upp_boud, event_low_boud = _df.Extinction.quantile([0.2, 0.8])
            _df_ext = _df.Extinction.copy()

            _df.loc[_df_ext >= event_low_boud, 'State'] = 'Event'
            _df.loc[(_df_ext < event_low_boud) & (_df_ext > clean_upp_boud), 'State'] = 'Transition'
            _df.loc[_df_ext <= clean_upp_boud, 'State'] = 'Clean'

            cond_event = _df.State == 'Event'
            cond_transition = _df.State == 'Transition'
            cond_clean = _df.State == 'Clean'

            dic_grp_sea[_grp] = {'Total': _df.copy(),
                                 'Clean': _df.loc[cond_clean].copy(),
                                 'Transition': _df.loc[cond_transition].copy(),
                                 'Event': _df.loc[cond_event].copy()}

        return dic_grp_sea


class StateClassifier(_Classifier):
    @classmethod
    def classify(cls, df) -> dict:
        df['Month'] = df.index.strftime('%Y-%m')
        df['Hour'] = df.index.hour
        df.loc[(df['Hour'] <= 18) & (df['Hour'] >= 7), 'Diurnal'] = 'Day'
        df.loc[(df['Hour'] <= 23) & (df['Hour'] >= 19), 'Diurnal'] = 'Night'
        df.loc[(df['Hour'] <= 6) & (df['Hour'] >= 0), 'Diurnal'] = 'Night'

        clean_upp_boud, event_low_boud = df.Extinction.quantile([0.2, 0.8])
        _df_ext = df.Extinction.copy()

        df.loc[_df_ext >= event_low_boud, 'State'] = 'Event'
        df.loc[(_df_ext < event_low_boud) & (_df_ext > clean_upp_boud), 'State'] = 'Transition'
        df.loc[_df_ext <= clean_upp_boud, 'State'] = 'Clean'

        cond_event = df.State == 'Event'
        cond_transition = df.State == 'Transition'
        cond_clean = df.State == 'Clean'

        dic_grp_sta = {'Total': df.copy(),
                       'Clean': df.loc[cond_clean].copy(),
                       'Transition': df.loc[cond_transition].copy(),
                       'Event': df.loc[cond_event].copy()}

        return dic_grp_sta


class Classifier:

    state = StateClassifier
    season = SeasonClassifier
    hour = HourClassifier

    def __new__(cls, df: DataFrame, clf: str):
        if clf == 'state':
            return cls.state(df)
        elif clf == 'season':
            return cls.season(df)
        else:
            return cls.hour(df)

