import numpy as np
from scipy.signal import find_peaks


def geometric(dp: np.ndarray, dist: np.ndarray, total):
    """ First change the distribution into dN """
    _dp = np.log(dp)
    _gmd = (((dist * _dp).sum()) / total.copy())

    _dp_mesh, _gmd_mesh = np.meshgrid(_dp, _gmd)
    _gsd = ((((_dp_mesh - _gmd_mesh) ** 2) * dist).sum() / total.copy()) ** .5

    return np.exp(_gmd), np.exp(_gsd)


def contribution(dp: np.ndarray, dist: np.ndarray, total):
    ultra_range = (dp >= 11.8) & (dp < 100)
    accum_range = (dp >= 100) & (dp < 1000)
    coars_range = (dp >= 1000) & (dp < 2500)

    _ultra = np.round(np.sum(dist[ultra_range]) / total, 2)
    _accum = np.round(np.sum(dist[accum_range]) / total, 2)
    _coars = np.round(np.sum(dist[coars_range]) / total, 2)

    return _ultra, _accum, _coars


def mode(dist: np.ndarray, **find_peaks_kwargs: dict):
    """ Find three peak mode in distribution. """

    min_value = np.array([min(dist)])
    extend_ser = np.concatenate([min_value, dist, min_value])
    _mode, _ = find_peaks(extend_ser, **dict(distance=len(dist)-1, **find_peaks_kwargs))

    return _mode - 1
