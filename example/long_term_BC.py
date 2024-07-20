import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from pandas import read_csv, concat

from DataPlot import *

TP_AE33_file_lst = ['/Users/chanchihyu/Downloads/TP_AE33_20180716-20191231.csv',
                    '/Users/chanchihyu/Downloads/TP_AE33_20200101-20211231.csv',
                    '/Users/chanchihyu/Downloads/TP_AE33_20220101-20221231.csv',
                    '/Users/chanchihyu/Downloads/TP_AE33_20230101-20230921.csv',
                    '/Users/chanchihyu/Downloads/TP_AE33_20231219-20240131.csv']

TP_BC1054_file_lst = ['/Users/chanchihyu/Downloads/TP_BC1054_20230927-1219.csv',
                      '/Users/chanchihyu/Downloads/TP_BC1054_20240214-0714.csv']

DH_AE33_file = '/Users/chanchihyu/Downloads/DH_AE33_2017-2022.csv'

FS_AE33_file = '/Users/chanchihyu/Downloads/FS_AE33_240301-0704.csv'

NZ_BC1054_file = '/Users/chanchihyu/Downloads/NZ_BC1054_240129-0622.csv'

resample = '1d'
rolling_window = 30

TP_AE33 = concat(
    [read_csv(file, index_col='time', parse_dates=['time']).resample(resample).mean() for file in TP_AE33_file_lst])

TP_BC1054 = concat(
    [read_csv(file, index_col='Time', parse_dates=['Time']).resample(resample).mean() for file in
     TP_BC1054_file_lst]).rename(columns={'BC9(ng/m3)': 'BC6'})

TP_AE33_1054 = TP_AE33.combine_first(TP_BC1054)
TP_AE33_1054 = TP_AE33_1054[~TP_AE33_1054.index.duplicated(keep='first')]

DH_AE33 = read_csv(DH_AE33_file, index_col='time', parse_dates=['time']).resample(resample).mean()

FS_AE33 = read_csv(FS_AE33_file, index_col='time', parse_dates=['time']).resample(resample).mean()

NZ_BC1054 = read_csv(NZ_BC1054_file, index_col='Time', parse_dates=['Time']).resample(resample).mean().rename(
    columns={'BC9 (ng/m3)': 'BC6'})


def rolling_tool(data, window=rolling_window):
    return data.rolling(window=window, min_periods=1).mean(), data.rolling(window=window, min_periods=1).std()


TP_AE33_1054_roll, TP_AE33_1054_std_roll = rolling_tool(TP_AE33_1054)
DH_AE33_roll, DH_AE33_std_roll = rolling_tool(DH_AE33)
FS_AE33_roll, FS_AE33_std_roll = rolling_tool(FS_AE33)
NZ_BC1054_roll, NZ_BC1054_std_roll = rolling_tool(NZ_BC1054)


@set_figure(figsize=(10, 5))
def plot():
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2, left=0.1, right=0.95)
    ax.scatter(TP_AE33_1054_roll.index, TP_AE33_1054_roll['BC6'], s=2, color='green', label='Taipei')
    ax.fill_between(TP_AE33_1054_roll.index, TP_AE33_1054_roll['BC6'] - TP_AE33_1054_std_roll['BC6'] * 0.5,
                    TP_AE33_1054_roll['BC6'] + TP_AE33_1054_std_roll['BC6'] * 0.5, color='green', alpha=0.2,
                    edgecolor=None)
    ax.scatter(DH_AE33_roll.index, DH_AE33_roll['BC'], s=2, color='blue', label='Taichung')
    ax.fill_between(DH_AE33_roll.index, DH_AE33_roll['BC'] - DH_AE33_roll['BC'] * 0.5,
                    DH_AE33_roll['BC'] + DH_AE33_roll['BC'] * 0.5, color='blue', alpha=0.2, edgecolor=None)
    ax.scatter(FS_AE33_roll.index, FS_AE33_roll['BC6'], s=2, color='r', label='Kaohsiung FS')
    ax.fill_between(FS_AE33_roll.index, FS_AE33_roll['BC6'] - FS_AE33_roll['BC6'] * 0.5,
                    FS_AE33_roll['BC6'] + FS_AE33_roll['BC6'] * 0.5, color='r', alpha=0.2, edgecolor=None)
    ax.scatter(NZ_BC1054_roll.index, NZ_BC1054_roll['BC6'], s=2, color='purple', label='Kaohsiung NZ')
    ax.fill_between(NZ_BC1054_roll.index, NZ_BC1054_roll['BC6'] - NZ_BC1054_roll['BC6'] * 0.5,
                    NZ_BC1054_roll['BC6'] + NZ_BC1054_roll['BC6'] * 0.5, color='purple', alpha=0.2, edgecolor=None)

    ax.set(xlim=(DH_AE33_roll.index[0], None),
           ylim=(0, None),
           ylabel='$BC\\ (ng/m^3)$',
           title='BC Long-term Observation')

    # Set major ticks locator to each month and formatter to year-month format
    ax.xaxis.set_major_locator(MonthLocator(bymonth=None, bymonthday=1, interval=6, tz=None))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    plt.xticks(rotation=30)

    ax.legend()


if __name__ == '__main__':
    plot()
