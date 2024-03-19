from datetime import datetime
from pandas import DataFrame


class Classifier:
    Seasons = {'2020-Summer': (datetime(2020, 9, 4), datetime(2020, 9, 21, 23)),
               '2020-Autumn': (datetime(2020, 9, 22), datetime(2020, 12, 29, 23)),
               '2020-Winter': (datetime(2020, 12, 30), datetime(2021, 3, 25, 23)),
               '2021-Spring': (datetime(2021, 3, 26), datetime(2021, 5, 6, 23))}
    # '2021-Summer': (datetime(2021, 5, 7), datetime(2021, 10, 16, 23))
    # '2021-Autumn': (datetime(2021, 10, 17), datetime(2021, 12, 31, 23))

    def __new__(cls, df):
        pass

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
