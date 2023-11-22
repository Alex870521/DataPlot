import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_csv, concat
from DataPlot.Data_processing.csv_decorator import save_to_csv
from DataPlot.Data_processing.Mie_plus import Mie_PESD, Mie_MEE

PATH_MAIN = Path(__file__).parent.parent.parent / 'Data' / 'Level2'
PATH_DIST = PATH_MAIN / 'distribution'

with open(PATH_DIST / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PNSD_dry.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD_dry = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_MAIN / 'mass_volume_VAM.csv', 'r', encoding='utf-8', errors='ignore') as f:
    refractive_index = read_csv(f, parse_dates=['Time']).set_index('Time')[['gRH', 'n_dry', 'n_amb', 'k_dry', 'k_amb', 'density']]

with open(PATH_MAIN / 'mass_volume_VAM.csv', 'r', encoding='utf-8', errors='ignore') as f:
    volume_ratio = read_csv(f, parse_dates=['Time']).set_index('Time')[['AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio', 'Soil_volume_ratio', 'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']]

df = concat([PNSD, refractive_index, volume_ratio], axis=1)

df_dry = concat([PNSD_dry, refractive_index, volume_ratio], axis=1)

dp = np.array(PNSD.columns, dtype='float')
_length = np.size(dp)
dlogdp = np.array([0.014] * _length)
index = df.index.copy()
df_input = df.dropna()


def selection(object1, object2, object3, kind):
    if kind == 'ext':
        return object1
    elif kind == 'sca':
        return object2
    else:
        return object3


def extinction_dist(column, mode, kind):
    if mode == 'internal':
        m = column['n_amb'] + 1j * column['k_amb']
        ndp = np.array(column[:_length])
        Ext_dist, Sca_dist, Abs_dist = Mie_PESD(m, 550, dp, dlogdp, ndp)
        output = selection(Ext_dist, Sca_dist, Abs_dist, kind=kind)

        return output

    if mode == 'external':
        RI_dic = {'AS': 1.53 + 0j,
                  'AN': 1.55 + 0j,
                  'OM': 1.54 + 0j,
                  'Soil': 1.56 + 0.01j,
                  'SS': 1.54 + 0j,
                  'BC': 1.80 + 0.54j,
                  'water': 1.333 + 0j}

        Ext_dist = np.zeros((167,))
        Sca_dist = np.zeros((167,))
        Abs_dist = np.zeros((167,))
        ndp = np.array(column[:_length])
        for _m, _specie in zip(RI_dic.values(),
                               ['AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio', 'Soil_volume_ratio',
                                'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']):
            _ndp = column[_specie] / (1 + column['ALWC_volume_ratio']) * ndp
            _Ext_dist, _Sca_dist, _Abs_dist = Mie_PESD(_m, 550, dp, dlogdp, _ndp)
            Ext_dist += _Ext_dist
            Sca_dist += _Sca_dist
            Abs_dist += _Abs_dist

        output = selection(Ext_dist, Sca_dist, Abs_dist, kind=kind)
        return np.array(output)


@save_to_csv(PATH_MAIN / 'PESD.csv')
def Extinction_PSD_process(data=df_input, reset=False, filename=None):
    """ Calculated the extinction distribution

    :param mode: 'mixing type'
    :param reset:
    :param filename:
    :return:
    """
    # 將輸入的data用extinction_dist計算"消光分布"
    Ext_dist = data.apply(extinction_dist, axis=1, args=('internal', 'ext'), result_type='expand').set_axis(dp, axis=1)
    Sca_dist = data.apply(extinction_dist, axis=1, args=('internal', 'sca'), result_type='expand').set_axis(dp, axis=1)
    Abs_dist = data.apply(extinction_dist, axis=1, args=('internal', 'abs'), result_type='expand').set_axis(dp, axis=1)

    Ext_dist2 = data.apply(extinction_dist, axis=1, args=('external', 'ext'), result_type='expand').set_axis(dp, axis=1)
    Sca_dist2 = data.apply(extinction_dist, axis=1, args=('external', 'sca'), result_type='expand').set_axis(dp, axis=1)
    Abs_dist2 = data.apply(extinction_dist, axis=1, args=('external', 'abs'), result_type='expand').set_axis(dp, axis=1)

    result_df = pd.DataFrame({'Bext_internal': Ext_dist.apply(np.sum, axis=1),
                              'Bsca_internal': Sca_dist.apply(np.sum, axis=1),
                              'Babs_internal': Abs_dist.apply(np.sum, axis=1),
                              'Bext_external': Ext_dist2.apply(np.sum, axis=1),
                              'Bsca_external': Sca_dist2.apply(np.sum, axis=1),
                              'Babs_external': Abs_dist2.apply(np.sum, axis=1),}).reindex(index)
    # GMD, GSD ...

    # Save .csv
    (Ext_dist/dlogdp).reindex(index).to_csv(PATH_DIST / f'PESD_dextdlogdp_internal.csv')
    (Ext_dist2/dlogdp).reindex(index).to_csv(PATH_DIST / f'PESD_dextdlogdp_external.csv')
    # Sca_dist.reindex(index).to_csv(PATH_DIST / f'PESD_dscadlogdp_{mode}.csv')
    # Abs_dist.reindex(index).to_csv(PATH_DIST / f'PESD_dabsdlogdp_{mode}.csv')

    return result_df


# internal mixing
@save_to_csv((PATH_MAIN / 'PESD_dry.csv', PATH_DIST / 'PESDist_dry.csv'))
def Extinction_dry_PSD_internal_process(reset=False, filename=None):
    index = df_dry.index.copy()
    df_input = df_dry.dropna()
    _index = df_input.index.copy()

    out = {'Bext': [],
           'Bsca': [],
           'Babs': [],
           }

    out_dist = {'Bext_dist': [],
                }

    for _tm, _ser in df_input.iterrows():
        ndp = np.array(_ser[:_length])
        _out, _out_dis = Mie_PESD(_ser['n_dry'] + 1j * _ser['k_dry'], 550, dp, dlogdp, ndp)
        for (key, out_lst) in out.items():
            out_lst.append(_out[key])
        for (key, out_lst) in out_dist.items():
            out_lst.append(_out_dis[key])

    Bext_df = pd.DataFrame(out).set_index(_index).reindex(index)
    Bext_df.rename(columns={'Bext': 'Bext_dry', 'Bsca': 'Bsca_dry', 'Babs': 'Babs_dry'}, inplace=True)
    Bext_dist_df = pd.DataFrame(out_dist['Bext_dist']).set_index(_index).set_axis(dp, axis=1).reindex(index)

    return Bext_df, Bext_dist_df


# external mixing
@save_to_csv((PATH_MAIN / 'PESD_dry_external.csv', PATH_DIST / 'PESDist_dry_external.csv'))
def Extinction_dry_PSD_external_process(reset=False, filename=None):
    index = df_dry.index.copy()
    df_input = df_dry.dropna()
    _index = df_input.index.copy()

    out = {'Bext': [],
           'Bsca': [],
           'Babs': [],
           }

    out_dist = {'Bext_dist': [],
                }

    for _tm, _ser in df_input.iterrows():
        ndp = np.array(_ser[:_length])

        # 7 species
        ndp1 = _ser['AS_volume_ratio'] * ndp
        ndp2 = _ser['AN_volume_ratio'] * ndp
        ndp3 = _ser['OM_volume_ratio'] * ndp
        ndp4 = _ser['Soil_volume_ratio'] * ndp
        ndp5 = _ser['SS_volume_ratio'] * ndp
        ndp6 = _ser['EC_volume_ratio'] * ndp

        # RI_dic = {'AS': 1.53 + 0j,
        #           'AN': 1.55 + 0j,
        #           'OM': 1.54 + 0j,
        #           'Soil': 1.56 + 0.01j,
        #           'SS': 1.54 + 0j,
        #           'BC': 1.80 + 0.54j,
        #           'water': 1.333 + 0j}

        AS_out, AS_out_dist = Mie_PESD(1.53 + 0j, 550, dp, dlogdp, ndp1)
        AN_out, AN_out_dist = Mie_PESD(1.55 + 0j, 550, dp, dlogdp, ndp2)
        OM_out, OM_out_dist = Mie_PESD(1.54 + 0j, 550, dp, dlogdp, ndp3)
        Soil_out, Soil_out_dist = Mie_PESD(1.56 + 0.01j, 550, dp, dlogdp, ndp4)
        SS_out, SS_out_dist = Mie_PESD(1.54 + 0j, 550, dp, dlogdp, ndp5)
        EC_out, EC_out_dist = Mie_PESD(1.80 + 0.54j, 550, dp, dlogdp, ndp6)

        for (key, out_lst) in out.items():
            out_lst.append(AS_out[key] + AN_out[key] + OM_out[key] + Soil_out[key] + SS_out[key] + EC_out[key])
        for (key, out_lst) in out_dist.items():
            out_lst.append(
                AS_out_dist[key] + AN_out_dist[key] + OM_out_dist[key] + Soil_out_dist[key] + SS_out_dist[key] +
                EC_out_dist[key])

    Bext_df = pd.DataFrame(out).set_index(_index).reindex(index)
    Bext_df.rename(columns={'Bext': 'Bext_dry_external', 'Bsca': 'Bsca_dry_external', 'Babs': 'Babs_dry_external'},
                   inplace=True)
    Bext_dist_df = pd.DataFrame(out_dist['Bext_dist']).set_index(_index).set_axis(dp, axis=1).reindex(index)

    return Bext_df, Bext_dist_df


if __name__ == '__main__':
    Extinction_PESD = Extinction_PSD_process(reset=True)
    # Extinction_dry_PSD_internal_process(reset=True)
    # Extinction_dry_PSD_external_process(reset=True)
