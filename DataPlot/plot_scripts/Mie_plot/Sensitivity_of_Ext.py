import numpy as np
import math
import pandas as pd
from pathlib import Path
from pandas import read_csv, concat
import pickle
from plot_templates.basic.scatter_plot import scatter
from processDecorator import function_handler
from Data_processing import Mie_PESD
from Data_processing import integrate
import matplotlib.pyplot as plt


PATH_MAIN = Path("/Data-Code-example")
PATH_DIST = Path("/Data-Code-example/Level2/distribution")

with open(PATH_DIST / 'PNSDist.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_MAIN / 'level2' / 'mass_volume_VAM.csv', 'r', encoding='utf-8', errors='ignore') as f:
    refractive_index = read_csv(f, parse_dates=['Time']).set_index('Time')[['gRH', 'n_dry', 'n_amb', 'k_dry', 'k_amb']]

with open(PATH_MAIN / 'All_data.csv', 'r', encoding='utf-8', errors='ignore') as f:
    All = read_csv(f, parse_dates=['Time'], low_memory=False).set_index('Time')

df = concat([PNSD, refractive_index], axis=1)

dp = np.array(PNSD.columns, dtype='float')
_length = np.size(dp)
dlogdp = np.array([0.014] * _length)


@function_handler
def Fixed_ext_process():
    _index = df.index.copy()
    df_input = df.dropna()

    out = {'Bext_Fixed_PNSD': [],
           'Bext_Fixed_RI': [],
           }

    Fixed_PNSD = np.array(df_input.iloc[:, :_length].mean())
    Fixed_n = np.array(df_input['n_amb'].mean())
    Fixed_k = np.array(df_input['k_amb'].mean())

    for _tm, _ser in df_input.iterrows():
        FixPNSD, _ = Mie_PESD(_ser['n_amb'] + 1j * _ser['k_amb'], 550, dp, dlogdp, ndp=Fixed_PNSD, output_dist=True)
        FixRI, __ = Mie_PESD(Fixed_n + 1j * Fixed_k, 550, dp, dlogdp, ndp=_ser[:_length], output_dist=True)
        out['Bext_Fixed_PNSD'].append(FixPNSD['Bext'])
        out['Bext_Fixed_RI'].append(FixRI['Bext'])

    Bext_df = pd.DataFrame(out).set_index(df_input.index.copy()).reindex(_index)
    return Bext_df


if __name__ == '__main__':
    # result = Fixed_ext_process()
    # with open(PATH_MAIN / 'fixed_PNSD_RI.pkl', 'wb') as f:
    #     pickle.dump(result, f)

    with open(PATH_MAIN / 'fixed_PNSD_RI.pkl', 'rb') as f:
        result = pickle.load(f)

    df = concat([result, All], axis=1)
    scatter(df, x='Extinction', y='Bext', xlim=[0, 600], ylim=[0, 600], title='Mie theory', regression=True, diagonal=True)
    scatter(df, x='Extinction', y='Bext_Fixed_PNSD', xlim=[0, 600], ylim=[0, 600], title='Fixed PNSD', regression=True, diagonal=True)
    scatter(df, x='Extinction', y='Bext_Fixed_RI', xlim=[0, 600], ylim=[0, 600], title='Fixed RI', regression=True, diagonal=True)

