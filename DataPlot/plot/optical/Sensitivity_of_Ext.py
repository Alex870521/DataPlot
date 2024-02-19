import numpy as np
from pathlib import Path
from pandas import concat, DataFrame
from DataPlot.plot import scatter
from DataPlot.process import DataReader, SizeDist, Mie_PESD


PATH_MAIN = Path(__file__).parents[0]

PNSD, RI = DataReader('PNSD_dNdlogdp.csv'), DataReader('chemical.csv')[['gRH', 'n_dry', 'n_amb', 'k_dry', 'k_amb']]

df = concat([PNSD, RI], axis=1)

dp = SizeDist().dp
_length = np.size(dp)
dlogdp = SizeDist().dlogdp


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
        FixPNSD = Mie_PESD(_ser['n_amb'] + 1j * _ser['k_amb'], 550, dp, dlogdp, ndp=Fixed_PNSD)
        FixRI = Mie_PESD(Fixed_n + 1j * Fixed_k, 550, dp, dlogdp, ndp=_ser[:_length])
        out['Bext_Fixed_PNSD'].append(FixPNSD['Bext'])
        out['Bext_Fixed_RI'].append(FixRI['Bext'])

    Bext_df = DataFrame(out).set_index(df_input.index.copy()).reindex(_index)
    return Bext_df


if __name__ == '__main__':
    result = Fixed_ext_process()

    # with open(PATH_MAIN / 'fixed_PNSD_RI.pkl', 'rb') as f:
    #     result = pickle.load(f)

    All = DataReader('All_data.csv')

    df = concat([result, All], axis=1)

    scatter(df, x='Extinction', y='Bext_internal', xlim=[0, 600], ylim=[0, 600], title='Mie theory', regression=True, diagonal=True)
    scatter(df, x='Extinction', y='Bext_Fixed_PNSD', xlim=[0, 600], ylim=[0, 600], title='Fixed PNSD', regression=True, diagonal=True)
    scatter(df, x='Extinction', y='Bext_Fixed_RI', xlim=[0, 600], ylim=[0, 600], title='Fixed RI', regression=True, diagonal=True)

