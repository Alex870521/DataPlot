import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from DataPlot import *


@set_figure(figsize=(5, 4), fs=6)
def plot_metal_heatmaps(df, iop):
    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to your data and transform it
    standardized_data = scaler.fit_transform(df)

    # Replace the original DataFrame columns with the standardized data
    df = pd.DataFrame(standardized_data, index=df.index, columns=df.columns)

    # Major ticks every 24 hours
    major_interval = 24
    major_tick_positions = range(0, len(df.index), major_interval)
    major_tick_labels = [df.index.strftime('%Y-%m-%d')[i] for i in major_tick_positions]

    # Minor ticks every 4 hours
    minor_interval = 4
    minor_tick_positions = range(0, len(df.index), minor_interval)

    # Plot the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df.T, cmap="jet", xticklabels=major_tick_labels, yticklabels=True, cbar_kws={'label': 'Concentration'})
    ax.grid(color='gray', linestyle='-', linewidth=0.3)

    # Set x-tick positions and labels
    ax.set_xticks(ticks=major_tick_positions, labels=major_tick_labels)
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.set_title(f"IOP {iop}", fontsize=10)
    ax.set(xlabel='', ylabel='')
    plt.tight_layout()


if __name__ == '__main__':
    IOP = {'1': ['2024-02-13', '2024-02-17'],
           '2': ['2024-02-26', '2024-03-01'],
           '3': ['2024-03-11', '2024-03-15'],
           '4': ['2024-03-25', '2024-03-29']}

    df = read_csv('/Users/chanchihyu/NTU/KSvis能見度計畫/FS/FS_chem.csv',
                  na_values=('???@????', '????????????', 'Nodata', '#', '*'), index_col='time', parse_dates=['time'])
    df.columns = df.columns.str.strip()
    df = df[['Al', 'Si', 'W', 'Cl', 'K', 'Sr', 'Ba', 'Bi', 'Cu', 'As', 'Ca',
             'Ti', 'V', 'Cr', 'Ni', 'Mn', 'Fe', 'Co', 'Zn', 'Ga', 'Ge',
             'Se', 'Br', 'Rb', 'Y', 'Zr', 'Nb', 'Mo', 'Pb', 'Pd', 'Ag', 'In', 'Te',
             'Cd', 'Sn', 'Sb', 'Cs', ]].resample('1h').mean()


    def remove_outliers(df):
        for column in df.columns:
            mean = df[column].mean()
            std = df[column].std()

            # Calculate the lower and upper bounds
            lower_bound = mean - 6 * std
            upper_bound = mean + 6 * std

            # Filter the data
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        return df


    def smooth_data(df):
        return df.rolling(window=2, min_periods=1).mean()


    df = smooth_data(df)
    df = remove_outliers(df).resample('1h').mean()
    df = df.interpolate(method='time')

    for iop, dates in IOP.items():
        plot_metal_heatmaps(df.loc[dates[0]:dates[1]], iop=iop)
        # plt.savefig(f'/Users/chanchihyu/NTU/KSvis能見度計畫/FS/heatmaps/IOP_{iop}.png')
