from pathlib import Path

import pandas as pd
from pandas import read_csv, concat
from tqdm import tqdm

from DataPlot import *


def pre_process():
    raw_folder = Path('/Users/chanchihyu/NTU/監資處資料/GRIMM_data/raw')
    output_folder = Path('/Users/chanchihyu/NTU/監資處資料/GRIMM_data/processed')
    output_folder.mkdir(parents=True, exist_ok=True)

    for folder in raw_folder.iterdir():
        if folder.name in ['.DS_Store']:
            continue

        print("Processing folder:", folder)

        lst = []
        process_bar = tqdm(list(folder.iterdir()), bar_format="{l_bar}{bar}|")
        for file in process_bar:
            if file.suffix == '.dat':
                process_bar.set_description(f"Processing file --> {file.name}")

                df = read_csv(file, header=233, delimiter='\t', index_col=0, parse_dates=[0]).rename_axis("Time")
                df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M:%S", dayfirst=True)

                if folder.name == "A407ST":
                    df.drop(df.columns[0:11].tolist() + df.columns[128:].tolist(), axis=1, inplace=True)
                else:
                    df.drop(df.columns[0:11].tolist() + df.columns[-5:].tolist(), axis=1, inplace=True)

                if df.empty:
                    print(file, "is empty")
                    continue

                lst.append(df)

        df = concat(lst).sort_index()
        st, en = df.index[0], df.index[-1]

        df.to_csv(output_folder / f'{folder.name}_{st.strftime("%F")}_{en.strftime("%F")}.csv')
        print("Finished processing folder:", folder)


def process():
    for file in Path('/Users/chanchihyu/GRIMM_data/processed').iterdir():
        if file.suffix == '.csv':
            print("Processing file:", file)

            dlogdp = 0.035
            df = read_csv(file, index_col='Time', parse_dates=True, dtype=float) / dlogdp

            df = df.resample('h').mean()

            def remove_outliers(df):
                Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
                IQR = Q3 - Q1
                return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

            data = remove_outliers(df)

            plot_dist(data)


def plot_dist(data: pd.DataFrame):
    plot.distribution.heatmap(data, unit='Number', magic_number=0)
    plot.distribution.heatmap_tms(data, unit='Number', freq='10d')


if __name__ == '__main__':
    pre_process()
    process()
