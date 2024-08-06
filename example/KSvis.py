from datetime import datetime

from pandas import concat, Timestamp
from tabulate import tabulate

from DataPlot import *

# NZ, FS = [], []
# NZ_file_lst = ['chem_NZ.csv', 'gas_NZ.csv', 'partition.csv']
# FS_file_lst = ['FS_chem.csv', 'FS_APS_PM.csv', 'partition.csv']
#
# for file in NZ_file_lst:
#     df = read_csv(f'/Users/chanchihyu/NTU/KSvis能見度計畫/NZ/data/{file}', parse_dates=['time'], index_col='time',
#                   na_values=('???@????', '????????????', 'Nodata', '#', '*', '-'))
#     NZ.append(df)
#
# for file in FS_file_lst:
#     df = read_csv(f'/Users/chanchihyu/NTU/KSvis能見度計畫/FS/data/{file}', parse_dates=['time'], index_col='time',
#                   na_values=('???@????', '????????????', 'Nodata', '#', '*', '-'))
#     FS.append(df)
#
# NZ = concat(NZ, axis=1).rename(columns=lambda x: x.strip())
# FS = concat(FS, axis=1).rename(columns=lambda x: x.strip())

IOP = {'1': ['2024-02-13', '2024-02-17'],
       '2': ['2024-02-26', '2024-03-01'],
       '3': ['2024-03-11', '2024-03-15'],
       '4': ['2024-03-25', '2024-03-29']}

df = read_csv('/Users/chanchihyu/NTU/KSvis能見度計畫/FS/data/FS_chem.csv',
              na_values=('???@????', '????????????', 'Nodata', '#', '*'), index_col='time', parse_dates=['time'])
df2 = read_csv('/Users/chanchihyu/NTU/KSvis能見度計畫/NZ/NZ 2~6月.csv',
               na_values=('???@????', '????????????', 'Nodata', '#', '*', '-'), index_col='time',
               parse_dates=['time'])

df.columns = df.columns.str.strip()
df2.columns = df2.columns.str.strip()

items = ['Al', 'Zr', 'Si', 'Ca', 'Ti', 'Mn', 'Fe', 'V', 'Cl', 'K',
         'Sr', 'Ba', 'Bi', 'Pd', 'Sn', 'Cr', 'W', 'Cu', 'Zn',
         'As', 'Co', 'Se', 'Br', 'Cd', 'Sb', 'In', 'Pb', 'Ni',
         ]

df = df[items].resample('1h').mean()
df2 = df2[items].resample('1h').mean()


def data_table(df: DataFrame,
               items: list[str] | str = None,
               times: tuple[datetime | Timestamp | str, datetime | Timestamp | str] = None,
               ):
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
    items = [items] if isinstance(items, str) else items
    times = [times] if isinstance(times, str) else times
    times = list(map(Timestamp, times))

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

    result = DataFrame(results, columns=items, index=[f'PERIOD {i + 1}' for i in range(len(times) + 1)])

    print(tabulate(result, headers='keys', tablefmt='fancy_grid'))


if __name__ == '__main__':
    # plot.meteorology.CBPF(NZ, 'WS', 'WD', 'NO', percentile=75)
    # plot.meteorology.CBPF(NZ, 'WS', 'WD', 'PM2.5', percentile=75)
    # plot.meteorology.CBPF(NZ, 'WS', 'WD', 'SO2', percentile=75)

    # plot.improve.ammonium_rich(NZ, title='NZ')
    # plot.improve.ammonium_rich(FS, title='FS')

    # data_table(NZ)

    # Concatenate df and df2 along the rows
    df = df.shift(freq='30min')
    combined_df = concat([df, df2])

    # Normalize the combined DataFrame
    normalized_combined_df = plot.process_data(combined_df)

    # Split the normalized DataFrame back into df and df2 using their original indices
    df = normalized_combined_df.loc[df.index]
    df2 = normalized_combined_df.loc[df2.index]

    df = df.shift(freq='-30min')

    for iop, dates in IOP.items():
        # plot_metal_heatmaps(df.loc[dates[0]:dates[1]], title=f'FS IOP {iop}')
        # plt.savefig('FS_IOP' + iop + '.png')
        plot.metal_heatmaps(df2.loc[dates[0]:dates[1]], title=f'NZ IOP {iop}')
        # plt.savefig('NZ_IOP' + iop + '.png')

    # plot.pie(data_set=data_NZ,
    #          labels=[rf'$AS$', rf'$AN$', rf'$OM$', rf'$EC$', rf'$Soil$', rf'$SS$'],
    #          unit='PM25',
    #          style='pie',
    #          )
