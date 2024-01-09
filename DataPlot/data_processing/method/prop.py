import numpy as np
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
    data = np.array(ser) * dlogdp
    _total = data.sum()

    ultra_range = (dp >= 11.8) & (dp < 100)
    accum_range = (dp >= 100)  & (dp < 1000)
    coars_range = (dp >= 1000) & (dp < 2500)

    _ultra = np.round(np.sum(data[ultra_range]) / _total, 2)
    _accum = np.round(np.sum(data[accum_range]) / _total, 2)
    _coars = np.round(np.sum(data[coars_range]) / _total, 2)

    return _ultra, _accum, _coars
