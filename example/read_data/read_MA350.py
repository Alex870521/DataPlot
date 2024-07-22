from datetime import datetime as dtm
from pathlib import Path

import pandas as pd

from DataPlot import *

start, end = dtm(2024, 5, 21), dtm(2024, 6, 28)

path_raw = Path('/Users/chanchihyu/NTU/台大貨櫃/MA350')
path_prcs = Path('prcs')


if __name__ == '__main__':
    df1 = RawDataReader('MA350', path_raw / 'MA350' / 'MA350_0171',
                        reset=True, start=start, end=end, mean_freq='1h', csv_out=True)
    df2 = RawDataReader('MA350', path_raw / 'MA350' / 'MA350_0176',
                        reset=True, start=start, end=end, mean_freq='1h', csv_out=True)

    df3 = read_csv(path_raw / 'TP_BC1054_20240214-0714.csv', index_col='Time',
                   parse_dates=['Time']).resample('1h').mean()

    df1.columns = ['MA350_0171 ' + col for col in df1.columns]
    df2.columns = ['MA350_0176 ' + col for col in df2.columns]
    df3.columns = ['BC1054 ' + col for col in df3.columns]

    _df = pd.concat([df1, df2, df3], axis=1)['2024-05-21': '2024-06-28']

    plot.scatter(_df, 'MA350_0171 BC1', 'BC1054 BC1(ng/m3)', xlim=(0, 3500), ylim=(0, 3500), regression=True,
                 diagonal=True)
    plot.scatter(_df, 'MA350_0171 BC2', 'BC1054 BC3(ng/m3)', xlim=(0, 3500), ylim=(0, 3500), regression=True,
                 diagonal=True)
    plot.scatter(_df, 'MA350_0171 BC3', 'BC1054 BC4(ng/m3)', xlim=(0, 3500), ylim=(0, 3500), regression=True,
                 diagonal=True)
    plot.scatter(_df, 'MA350_0171 BC5', 'BC1054 BC9(ng/m3)', xlim=(0, 3500), ylim=(0, 3500), regression=True,
                 diagonal=True)


    def rolling_tool(data, window=12):
        return data.rolling(window=window, min_periods=1).mean(), data.rolling(window=window, min_periods=1).std()
