from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from pandas import read_csv, concat, DataFrame
from tqdm import tqdm

from .core import DataReader, Classifier
from .script import ImpactProcessor, ImproveProcessor, ChemicalProcessor, ParticleSizeDistProcessor, OthersProcessor

__all__ = ['DataBase',
           'DataReader',
           'DataClassifier',
           'ParticleSizeDistProcessor'
           ]


class MainProcessor:
    def __init__(self, reset=False, filename='All_data.csv'):
        self.reset = reset
        self.file_path = Path(__file__).parents[1] / 'data' / filename

    def process_data(self):
        with tqdm(total=20, desc="Loading Data", bar_format="{l_bar}{bar}|", unit="it") as progress_bar:

            if self.file_path.exists() and not self.reset:
                progress_bar.update(20)
                return read_csv(self.file_path, parse_dates=['Time'], na_values=('-', 'E', 'F'),
                                low_memory=False).set_index('Time')

            else:
                processor = [ImpactProcessor, ChemicalProcessor, ImproveProcessor, ParticleSizeDistProcessor]
                reset = [False, False, False, False, False]
                save_filename = ['IMPACT.csv', 'chemical.csv', 'revised_IMPROVE.csv', 'PSD.csv', 'PESD.csv']

                _df = concat([processor().process_data(reset, save_filename) for processor, reset, save_filename in
                              zip(processor, reset, save_filename)])
                # 1. EPB
                minion = DataReader('EPB.csv')

                # 2. IMPACT
                impact = ImpactProcessor().process_data(reset=False)

                # 3. Mass_volume
                chemical = ChemicalProcessor().process_data(reset=False)

                # 4. improve
                improve = ImproveProcessor().process_data(reset=False, save_filename='revised_IMPROVE.csv',
                                                          version='revised')

                # 5. Number & Surface & volume & Extinction distribution
                psd = ParticleSizeDistProcessor().process_data(reset=False)

                _df = concat([minion, impact, chemical, improve, psd], axis=1)

                # 6. others
                _df = OthersProcessor(reset=False, data=_df).process_data()
                progress_bar.update(20)

                # 7. save result
                _df.to_csv(self.file_path)

                return _df.copy()


DataBase = MainProcessor(reset=False).process_data()


class DataClassifier(Classifier):
    """
    Notes
    -----
    First, create group then return the selected statistic method.
    If the 'by' does not exist in DataFrame, import the default DataFrame to help to sign the different group.

    """
    def __new__(cls, df: DataFrame,
                by: Literal["Hour", "State", "Season", "Season_state"] | str,
                statistic: Literal["Table", "Array", "Series", "Dict"] = 'Array',
                cut_bins: Sequence = None,
                qcut: int = None,
                labels: list[str] = None
                ):
        # group data
        if by not in df.columns:
            default = cls.classify(DataBase)
            df = concat([df, default[f'{by}']], axis=1)

        if cut_bins is not None:
            df = df.copy()
            midpoint = (cut_bins + (cut_bins[1] - cut_bins[0]) / 2)[:-1]
            df.loc[:, f'{by}_cut'] = pd.cut(df.loc[:, f'{by}'], cut_bins, labels=labels or midpoint)
            df = df.drop(columns=[f'{by}'])
            group = df.groupby(f'{by}_cut', observed=False)

        elif qcut is not None:
            df = df.copy()
            df.loc[:, f'{by}_qcut'] = pd.qcut(df.loc[:, f'{by}'], q=qcut, labels=labels)
            df = df.drop(columns=[f'{by}'])
            group = df.groupby(f'{by}_qcut', observed=False)

        else:
            if by == 'State':
                group = df.groupby(pd.Categorical(df['State'], categories=['Clean', 'Transition', 'Event']), observed=False)
            elif by == 'Season':
                group = df.groupby(pd.Categorical(df['Season'], categories=['2020-Summer', '2020-Autumn', '2020-Winter', '2021-Spring']), observed=False)
            else:
                group = df.groupby(f'{by}', observed=False)

        # return data
        if statistic == 'Array':
            return cls.returnArray(group)
        elif statistic == 'Table':
            return cls.returnTable(group)
        elif statistic == 'Series':
            return cls.returnSeries(group)
        else:
            return cls.returnDict(df, group)

    @staticmethod
    def returnArray(group):  # distribution ues
        _avg, _std = {}, {}

        for _grp, _df in group:
            _avg[_grp] = np.array(_df.mean(numeric_only=True))
            _std[_grp] = np.array(_df.std(numeric_only=True))

        return _avg, _std

    @staticmethod
    def returnSeries(group):
        _avg, _std = {}, {}

        for _grp, _df in group:
            _avg[_grp] = _df.mean(numeric_only=True)
            _std[_grp] = _df.std(numeric_only=True)

        return _avg, _std

    @staticmethod
    def returnTable(group) -> tuple[pd.DataFrame, pd.DataFrame]:
        return group.mean(numeric_only=True), group.std(numeric_only=True)

    @classmethod
    def returnDict(cls, df, group):
        dic_grp = {'Total': df}
        for _grp, _df in group:
            dic_grp[_grp] = _df.copy()

        return dic_grp
