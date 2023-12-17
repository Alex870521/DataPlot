import numpy as np
import math
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks


def geometric(dp, dlogdp, ser):
    """ First change the distribution into dN """
    num = np.array(ser) * dlogdp
    total_num = num.sum()

    _dp = np.log(dp)
    _gmd = (((num * _dp).sum()) / total_num.copy())

    _dp_mesh, _gmd_mesh = np.meshgrid(_dp, _gmd)
    _gsd = ((((_dp_mesh - _gmd_mesh) ** 2) * num).sum() / total_num.copy()) ** .5

    return np.exp(_gmd), np.exp(_gsd)


def peak_mode(dp, ser):
    # 3 mode
    min_value = np.array([min(ser)])
    extend_ser = np.concatenate([min_value, ser, min_value])
    _mode, _ = find_peaks(extend_ser, distance=20)
    return dp[_mode - 1][:3]


def mode_cont(dp, dlogdp, ser):
    num = np.array(ser) * dlogdp
    total_num = num.sum()

    ultra_range  = (dp >= 11.8) & (dp < 100)
    accum_range  = (dp >= 100)  & (dp < 1000)
    coarse_range = (dp >= 1000) & (dp < 2500)

    ultra_num = np.sum(num[ultra_range])
    accum_num = np.sum(num[accum_range])
    coars_num = np.sum(num[coarse_range])

    return [np.round(ultra_num/total_num, 2),
            np.round(accum_num/total_num, 2),
            np.round(coars_num/total_num, 2)]
