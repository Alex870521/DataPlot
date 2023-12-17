import numpy as np
import math
import pandas as pd
from pathlib import Path
from DataPlot.data_processing.Mie_theory import Mie_PESD


def internal_ext_dist(dp, dlogdp, ser):
    m = ser['n_amb'] + 1j * ser['k_amb']
    ndp = np.array(ser[:np.size(dp)])
    ext_dist, sca_dist, abs_dist = Mie_PESD(m, 550, dp, dlogdp, ndp)
    return dict(ext=ext_dist, sca=sca_dist, abs=abs_dist)


def external_ext_dist(dp, dlogdp, ser):
    refractive_dic = {'AS': 1.53 + 0j,
                      'AN': 1.55 + 0j,
                      'OM': 1.54 + 0j,
                      'Soil': 1.56 + 0.01j,
                      'SS': 1.54 + 0j,
                      'BC': 1.80 + 0.54j,
                      'water': 1.333 + 0j}

    ext_dist, sca_dist, abs_dist = np.zeros((167,)), np.zeros((167,)), np.zeros((167,))
    ndp = np.array(ser[:np.size(dp)])

    for _m, _specie in zip(refractive_dic.values(),
                           ['AS_volume_ratio', 'AN_volume_ratio', 'OM_volume_ratio', 'Soil_volume_ratio',
                            'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']):
        _ndp = ser[_specie] / (1 + ser['ALWC_volume_ratio']) * ndp
        _Ext_dist, _Sca_dist, _Abs_dist = Mie_PESD(_m, 550, dp, dlogdp, _ndp)
        ext_dist += _Ext_dist
        sca_dist += _Sca_dist
        abs_dist += _Abs_dist

    return dict(ext=ext_dist, sca=sca_dist, abs=abs_dist)