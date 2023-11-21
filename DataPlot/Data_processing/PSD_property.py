import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
from pandas import read_csv, concat
from processDecorator import save_to_csv

PATH_MAIN = Path(__file__).parent.parent / 'Data' / 'Level2'
PATH_DIST = PATH_MAIN / 'distribution'

with open(PATH_DIST / 'PNSDist.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PNSDist_dry.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD_dry = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PESDist.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PESD = read_csv(f, parse_dates=['Time']).set_index('Time')


dp = np.array(PNSD.columns, dtype='float')
_length = np.size(dp)
dlogdp = np.array([0.014] * _length)


def dist_prop(dist):
    peaks1, _ = find_peaks(np.concatenate(([min(dist)], dist, [min(dist)])), distance=20)
    num = np.array(dist * dlogdp)
    total_num = np.sum(num)

    ultra_num = np.sum(num[0:67]).__round__(4)
    accum_num = np.sum(num[67:139]).__round__(4)
    PM1_num = (ultra_num + accum_num).__round__(4)
    coars_num = np.sum(num[139:167]).__round__(4)

    GMD, GSD = geometric_prop(dp, num)

    contrbution = [(ultra_num / total_num).__round__(4), (accum_num / total_num).__round__(4), (coars_num / total_num).__round__(4)]
    return dict(mode=dp[peaks1-1], GMD=GMD, GSD=GSD, PM1_num=PM1_num, PM25_num=total_num, contrbution=contrbution, )


def geometric_prop(_dp, _prop):
    _prop_t = _prop.sum()

    _dp = np.log(_dp)
    _gmd = (((_prop * _dp).sum()) / _prop_t.copy())

    _dp_mesh, _gmd_mesh = np.meshgrid(_dp, _gmd)
    _gsd = ((((_dp_mesh - _gmd_mesh) ** 2) * _prop).sum() / _prop_t.copy()) ** .5

    return np.exp(_gmd).__round__(2), np.exp(_gsd).__round__(2)


dist = PESD.dropna().iloc[1]
aaa = dist_prop(dist)
