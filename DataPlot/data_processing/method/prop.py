import numpy as np
from scipy.signal import find_peaks


def geometric(dp, _dist, _total):
    """ First change the distribution into dN """
    _dp = np.log(dp)
    _gmd = (((_dist * _dp).sum()) / _total.copy())

    _dp_mesh, _gmd_mesh = np.meshgrid(_dp, _gmd)
    _gsd = ((((_dp_mesh - _gmd_mesh) ** 2) * _dist).sum() / _total.copy()) ** .5

    return np.exp(_gmd), np.exp(_gsd)


def mode(dp, _dist):
    """ Find three peak mode in distribution.

    Parameters
    ----------
    dp
    _dist

    Returns
    -------

    """
    min_value = np.array([min(_dist)])
    extend_ser = np.concatenate([min_value, _dist, min_value])
    _mode, _ = find_peaks(extend_ser, distance=20)

    return dp[_mode - 1][:3]


def contribution(dp, _dist, _total):
    ultra_range = (dp >= 11.8) & (dp < 100)
    accum_range = (dp >= 100)  & (dp < 1000)
    coars_range = (dp >= 1000) & (dp < 2500)

    _ultra = np.round(np.sum(_dist[ultra_range]) / _total, 2)
    _accum = np.round(np.sum(_dist[accum_range]) / _total, 2)
    _coars = np.round(np.sum(_dist[coars_range]) / _total, 2)

    return _ultra, _accum, _coars
