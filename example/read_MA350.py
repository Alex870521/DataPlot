from datetime import datetime as dtm
from pathlib import Path

from pandas import read_csv, concat

from DataPlot import *
from DataPlot.rawDataReader import *

start = dtm(2024, 5, 21)
end = dtm(2024, 6, 28)

path_raw = Path('/Users/chanchihyu/data')
path_prcs = Path('prcs')

if __name__ == '__main__':
    df1 = MA350.Reader(path_raw / 'MA350' / 'MA350_0171', reset=True)(start, end, mean_freq='1h', csv_out=True)
    df2 = MA350.Reader(path_raw / 'MA350' / 'MA350_0176', reset=True)(start, end, mean_freq='1h', csv_out=True)
    df3 = read_csv('/Users/chanchihyu/Downloads/TP_BC1054_20240214-0714.csv', index_col='Time',
                   parse_dates=['Time']).resample('1h').mean()
    df1.columns = ['MA350_0171 ' + col for col in df1.columns]
    df2.columns = ['MA350_0176 ' + col for col in df2.columns]
    df3.columns = ['BC1054 ' + col for col in df3.columns]

    # df = df.rename(columns={'BC1': 'MA350_0171 IR BC1', 'BC5': 'MA350_0171 IR BCc'})
    # df2 = df2.rename(columns={'BC1': 'MA350_0176 IR BC1', 'BC5': 'MA350_0176 IR BCc'})
    # df3 = df3.rename(columns={'BC1(ng/m3)': 'BC1054 BC1', 'BC9(ng/m3)': 'BC1054 IR BCc'})

    _df = concat([df1, df2, df3], axis=1)['2024-05-21': '2024-06-28']
    # plot.scatter(_df, 'MA350_0171 IR BCc', 'MA350_0176 IR BCc', xlim=(0, 3500), ylim=(0, 3500), regression=True, diagonal=True)
    # plot.scatter(_df, 'MA350_0171 IR BCc', 'BC1054 IR BCc', xlim=(0, 3500), ylim=(0, 3500), regression=True, diagonal=True)
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

    # _df_roll, _df_std_roll = rolling_tool(_df)
    # plot.optical.plot_MA350(_df_roll)
    # plot.optical.plot_day_night(_df)
