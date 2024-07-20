from typing import Union

import pandas as pd
from pandas import read_csv, concat, DataFrame, Timestamp
from tabulate import tabulate

NZ, FS = [], []
NZ_file_lst = ['chem_NZ.csv', 'gas_NZ.csv', 'partition.csv']
FS_file_lst = ['FS_chem.csv', 'FS_APS_PM.csv', 'partition.csv']

for file in NZ_file_lst:
    df = read_csv(f'/Users/chanchihyu/NTU/KSvis能見度計畫/NZ/{file}', parse_dates=['time'], index_col='time',
                  na_values=('Nodata'))
    NZ.append(df)

for file in FS_file_lst:
    df = read_csv(f'/Users/chanchihyu/NTU/KSvis能見度計畫/FS/{file}', parse_dates=['time'], index_col='time',
                  na_values=('Nodata'))
    FS.append(df)

NZ = concat(NZ, axis=1).rename(columns=lambda x: x.strip())
FS = concat(FS, axis=1).rename(columns=lambda x: x.strip())


def data_table(df: DataFrame,
               items: list[str] | str = ['NO', 'NO2', 'NOx'],
               times: Union[list[Union[str, Timestamp]], Timestamp, str] = ['2024-03-21', '2024-04-30']):
    """
    This function cuts the DataFrame based on the given time periods and calculates the mean and standard deviation
    of the specified items for each period.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be processed. It should have a DateTime index.
    items : list[str] | str, optional
        The columns of the DataFrame to be processed. It can be a list of column names or a single column name.
        By default, it is ['NO', 'NO2', 'NOx'].
    times : list[str] | str, optional
        The time periods to cut the DataFrame. It can be a list of time strings or a single time string.
        Each time string should be in the format of 'YYYY-MM-DD'. By default, it is ['2024-03-21', '2024-04-30'].

    Returns
    -------
    None
        This function doesn't return any value. It prints out a table showing the mean and standard deviation
        of the specified items for each time period.
    """
    if isinstance(items, str):
        items = [items]

    if isinstance(times, str):
        times = [times]

    if not isinstance(times, Timestamp or list[Timestamp]):
        times = [Timestamp(t) for t in times]

    times.sort()

    results = []
    for i in range(len(times) + 1):
        if i == 0:
            df_period = df.loc[df.index <= times[i], items]
        elif i == len(times):
            df_period = df.loc[df.index > times[i - 1], items]
        else:
            df_period = df.loc[(df.index > times[i - 1]) & (df.index <= times[i]), items]

        mean, std = df_period.mean().round(2).to_numpy(), df_period.std().round(2).to_numpy()

        results.append([f'{m} ± {s}' for m, s in zip(mean, std)])

    result = pd.DataFrame(results, columns=items, index=[f'PERIOD {i + 1}' for i in range(len(times) + 1)])

    print(tabulate(result, headers='keys', tablefmt='fancy_grid'))


if __name__ == '__main__':
    # plot.meteorology.CBPF(NZ, 'WS', 'WD', 'NO', percentile=75)
    # plot.meteorology.CBPF(NZ, 'WS', 'WD', 'PM2.5', percentile=75)
    # plot.meteorology.CBPF(NZ, 'WS', 'WD', 'SO2', percentile=75)

    # plot.meteorology.CBPF(df, 'WS', 'WD', 'NOx', percentile=75)
    # plot.meteorology.CBPF(df, 'WS', 'WD', 'SOR')

    # plot.improve.ammonium_rich(NZ, title='NZ')
    # plot.improve.ammonium_rich(FS, title='FS')

    data_table(NZ)

    data_NZ = {'Before': [19.2, 22.8, 42.7, 3.5, 8.6, 3.2], 'After': [24.4, 13.5, 44.4, 4.1, 10.9, 2.7]}
    data_FS = {'Before': [21.3, 31.6, 27.5, 3.7, 12.5, 3.4]}

    # plot.pie(data_set=data_NZ,
    #          labels=[rf'$AS$', rf'$AN$', rf'$OM$', rf'$EC$', rf'$Soil$', rf'$SS$'],
    #          unit='PM25',
    #          style='pie',
    #          )
