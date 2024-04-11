import numpy as np
from scipy.signal import find_peaks


def geometric(dp: np.ndarray, dist: np.ndarray, total):
    """ First change the distribution into dN """
    logdp = np.log(dp)
    _gmd = (((dist * logdp).sum()) / total.copy())

    dp_mesh, gmd_mesh = np.meshgrid(logdp, _gmd)
    _gsd = ((((dp_mesh - gmd_mesh) ** 2) * dist).sum() / total.copy()) ** .5

    return np.exp(_gmd), np.exp(_gsd)


def contribution(dp: np.ndarray, dist: np.ndarray, total):
    ultra_range = (dp >= 11.8) & (dp < 100)
    accum_range = (dp >= 100) & (dp < 1000)
    coars_range = (dp >= 1000) & (dp < 2500)

    ultra = np.round(np.sum(dist[ultra_range]) / total, 2)
    accum = np.round(np.sum(dist[accum_range]) / total, 2)
    coars = np.round(np.sum(dist[coars_range]) / total, 2)

    return ultra, accum, coars


def mode(dist: np.ndarray, **find_peaks_kwargs: dict):
    """ Find three peak mode in distribution. """

    min_value = np.array([min(dist)])
    extend_ser = np.concatenate([min_value, dist, min_value])
    _mode, _ = find_peaks(extend_ser, distance=len(dist) - 1, **find_peaks_kwargs)

    return _mode - 1
