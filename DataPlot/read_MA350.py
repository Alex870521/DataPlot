from pathlib import Path

from pandas import read_csv
from tqdm import tqdm

from DataPlot import *

raw_folder = Path('/Users/chanchihyu/NTU/MA350/raw')
output_folder = Path('/Users/chanchihyu/NTU/MA350/processed')
output_folder.mkdir(parents=True, exist_ok=True)


def pre_process():
    progress_bar = tqdm(list(raw_folder.iterdir()), bar_format="{l_bar}{bar}|")
    for file in progress_bar:
        if file.suffix == '.csv':
            progress_bar.set_description(f"Processing file --> {file.name}")

            df = read_csv(file, parse_dates=["Date / time local"], index_col="Date / time local").rename_axis("Time")

            df = df.sort_index()
            st, en = df.index[0], df.index[-1]

            df.to_csv(output_folder / f'MA350_{st.strftime("%F")}_{en.strftime("%F")}.csv')


def process():
    progress_bar = tqdm(list(output_folder.iterdir()), bar_format="{l_bar}{bar}|")
    for file in progress_bar:
        if file.suffix == '.csv':
            progress_bar.set_description(f"Processing file --> {file.name}")

            items = [
                "UV BCc",
                "Blue BCc",
                "Green BCc",
                "Red BCc",
                "IR BCc",
                "Biomass BCc  (ng/m^3)",
                "Fossil fuel BCc  (ng/m^3)",
                "AAE",
                "BB (%)"
            ]
            df = read_csv(file, index_col='Time', parse_dates=['Time'])[items].rename(
                columns={
                    "Biomass BCc  (ng/m^3)": "Biomass",
                    "Fossil fuel BCc  (ng/m^3)": "Fossil",
                    "BB (%)": "BB"
                }
            )

            def remove_outliers(df):
                Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
                IQR = Q3 - Q1
                return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

            df = remove_outliers(df)
            df['UV abs'] = df['UV BCc'] * 18.47 / 1000
            df['Blue abs'] = df['Blue BCc'] * 14.54 / 1000
            df['Green abs'] = df['Green BCc'] * 13.14 / 1000
            df['Red abs'] = df['Red BCc'] * 10.50 / 1000
            df['IR abs'] = df['IR BCc'] * 7.77 / 1000

            plot.optical.plot_MA350(df)
            plot.optical.plot_MA3502(df)
            plot.optical.plot_Bimass_Fossil(df)
            break


if __name__ == '__main__':
    # pre_process()
    process()
