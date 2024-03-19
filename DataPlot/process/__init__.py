import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Sequence
from pandas import read_csv, concat, DataFrame
from tqdm import tqdm
from collections import OrderedDict
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
        with tqdm(total=20, desc="Loading Data", bar_format="{l_bar}{bar}|", unit="it") as progress_bar:

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
