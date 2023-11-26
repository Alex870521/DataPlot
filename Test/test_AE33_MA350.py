import pandas as pd
from pathlib import Path
from pandas import read_table
from DataPlot.plot_templates import set_figure, scatter

PATH_MAIN = Path(__file__).parent


def read_AE33():
    with open(PATH_MAIN / 'AE33_AE33-S05-00503_20231122.dat', 'r', encoding='utf-8', errors='ignore') as f:
        _df = read_table(f, parse_dates={'Time': [0, 1]}, index_col='Time',
                         delimiter='\s+', skiprows=8, names=range(71))

        columns = ['Timebase', 'RefCh1', 'Sen1Ch1', 'Sen2Ch1', 'RefCh2', 'Sen1Ch2',
                   'Sen2Ch2', 'RefCh3', 'Sen1Ch3', 'Sen2Ch3', 'RefCh4', 'Sen1Ch4', 'Sen2Ch4', 'RefCh5', 'Sen1Ch5',
                   'Sen2Ch5',
                   'RefCh6', 'Sen1Ch6', 'Sen2Ch6', 'RefCh7', 'Sen1Ch7', 'Sen2Ch7', 'Flow1', 'Flow2', 'FlowC',
                   'Pressure (Pa)',
                   'Temperature (°C)', 'BB (%)', 'ContTemp', 'SupplyTemp', 'Status', 'ContStatus', 'DetectStatus',
                   'LedStatus',
                   'ValveStatus', 'LedTemp', 'BC11', 'BC12', 'BC1', 'BC21', 'BC22', 'BC2', 'BC31', 'BC32', 'BC3', 'BC41',
                   'BC42',
                   'BC4', 'BC51', 'BC52', 'BC5', 'BC61', 'BC62', 'BC6', 'BC71', 'BC72', 'BC7', 'K1', 'K2', 'K3', 'K4', 'K5',
                   'K6', 'K7',
                   'TapeAdvCount', 'ID_com1', 'ID_com2', 'ID_com3', 'fields_i']

        _df.columns = columns
        _df = _df[['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'Status']]
        _status = _df['Status'].copy()

        ## remove data without Status=0, 128(Not much filter tape), 256(Not much filter tape)
        _df = _df.where((_df['Status'] == 0) | (_df['Status'] == 128) | (_df['Status'] == 256)).copy()
        _df['Status'] = _status

    _df = _df[['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7']].mask((_df < 0).copy())

    ## call by _QC function
    ## QC data in 1 hr
    def _QC_func(_df_1hr):
        _df_ave = _df_1hr.mean()
        _df_std = _df_1hr.std()
        _df_lowb, _df_highb = _df_1hr < (_df_ave - _df_std * 1.5), _df_1hr > (_df_ave + _df_std * 1.5)

        return _df_1hr.mask(_df_lowb | _df_highb).copy()

    return _df.resample('1h', group_keys=False).apply(_QC_func).resample('1T').mean()


df1 = read_AE33().dropna()


def read_MA350():
    with open(PATH_MAIN / 'MA350-0045_S0012_231122095100.csv', 'r', encoding='utf-8', errors='ignore') as f:
        _df = pd.read_csv(f, delimiter=',', parse_dates={'Time': ['Date local (yyyy/MM/dd)', 'Time local (hh:mm:ss)']})

    _df = _df[['Time', 'UV BCc', 'Blue BCc', 'Green BCc', 'Red BCc', 'IR BCc']].set_index('Time')
    return _df


df2 = read_MA350().dropna()

df_ = pd.concat([df1, df2], axis=1)

scatter(df_, x='BC6', y='IR BCc', regression=True, title='AE33 vs. MA350')
