import numpy as np
import math
import pandas as pd

from pathlib import Path
from pandas import read_csv
from DataPlot.Data_processing.csv_decorator import save_to_csv

PATH_MAIN = Path(__file__).parent.parent.parent / 'Data' / 'Level2'
PATH_DIST = PATH_MAIN / 'distribution'


class DataTypeError(Exception):
    """ 請確認粒徑分布的數據 """

def number_dist(column):
    """ make sure the input data unit is dN/dlogdp """
    return np.array(column) * dlogdp


def surface_dist(column):
    """ make sure the input data unit is dN/dlogdp """
    return math.pi * (dp ** 2) * np.array(column) * dlogdp


def volume_dist(column):
    """ make sure the input data unit is dN/dlogdp """
    return math.pi / 6 * dp ** 3 * np.array(column) * dlogdp


def geometric_prop(column):
    num = np.array(column)
    total_num = num.sum()

    _dp = np.log(dp)
    _gmd = (((num * _dp).sum()) / total_num.copy())

    _dp_mesh, _gmd_mesh = np.meshgrid(_dp, _gmd)
    _gsd = ((((_dp_mesh - _gmd_mesh) ** 2) * num).sum() / total_num.copy()) ** .5

    return np.exp(_gmd), np.exp(_gsd)


@save_to_csv(PATH_MAIN / 'PNSD.csv')
def Number_PSD_process(data, reset=False, **kwargs):
    """ """
    num_dist = data.apply(number_dist, axis=1, result_type='broadcast')
    num_prop = num_dist.apply(geometric_prop, axis=1, result_type='expand')

    num_df = pd.DataFrame({'Number': num_dist.apply(np.sum, axis=1),
                           'GMDn': num_prop[0],
                           'GSDn': num_prop[1]})

    return num_df.reindex(index)


@save_to_csv(PATH_MAIN / 'PSSD.csv')
def Surface_PSD_process(data, reset=False, **kwargs):
    # Distribution
    surf_dist = data.apply(surface_dist, axis=1, result_type='broadcast')

    # GMD, GSD ...
    surf_prop = surf_dist.apply(geometric_prop, axis=1, result_type='expand')

    # Output data
    surf_df = pd.DataFrame({'Surface': surf_dist.apply(np.sum, axis=1),
                            'GMDs': surf_prop[0],
                            'GSDs': surf_prop[1]})

    # Save .csv
    (surf_dist / dlogdp).reindex(index).to_csv(PATH_DIST / 'PSSD_dSdlogdp.csv')
    return surf_df.reindex(index)


@save_to_csv(PATH_MAIN / 'PVSD.csv')
def Volume_PSD_process(data, reset=False, **kwargs):
    vol_dist = data.apply(volume_dist, axis=1, result_type='broadcast')

    # GMD, GSD ...
    vol_prop = vol_dist.apply(geometric_prop, axis=1, result_type='expand')

    vol_df = pd.DataFrame({'Volume': vol_dist.apply(np.sum, axis=1),
                           'GMDv': vol_prop[0],
                           'GSDv': vol_prop[1]})

    # Save .csv
    (vol_dist / dlogdp).reindex(index).to_csv(PATH_DIST / 'PVSD_dVdlogdp.csv')
    return vol_df.reindex(index)


if __name__ == '__main__':
    with open(PATH_DIST / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    # with open(PATH_DIST / 'PNSD_dry.csv', 'r', encoding='utf-8', errors='ignore') as f:
    #     PNSD_dry = read_csv(f, parse_dates=['Time']).set_index('Time')

    # basic parameter
    dp = np.array(PNSD.columns, dtype='float')
    _length = np.size(dp)
    dlogdp = np.array([0.014] * _length)
    index = PNSD.index.copy()
    df_input = PNSD.dropna()


    Number_PNSD = Number_PSD_process(reset=True)
    # Surface_PNSD = Surface_PSD_process(reset=True)
    # Volume_PNSD = Volume_PSD_process(reset=True)
