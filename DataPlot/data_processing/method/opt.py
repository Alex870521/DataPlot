import numpy as np
from DataPlot.data_processing.Mie_theory import Mie_PESD


def internal(ser, dp, dlogdp, wavelength=550):
    m = ser['n_amb'] + 1j * ser['k_amb']
    ndp = np.array(ser[:np.size(dp)])
    ext_dist, sca_dist, abs_dist = Mie_PESD(m, wavelength, dp, dlogdp, ndp)

    return dict(ext=ext_dist, sca=sca_dist, abs=abs_dist)


def external(ser, dp, dlogdp, wavelength=550):
    refractive_dic = {'AS_volume_ratio':   1.53 + 0j,
                      'AN_volume_ratio':   1.55 + 0j,
                      'OM_volume_ratio':   1.54 + 0j,
                      'Soil_volume_ratio': 1.56 + 0.01j,
                      'SS_volume_ratio':   1.54 + 0j,
                      'EC_volume_ratio':   1.80 + 0.54j,
                      'ALWC_volume_ratio': 1.333 + 0j}

    ext_dist, sca_dist, abs_dist = np.zeros((167,)), np.zeros((167,)), np.zeros((167,))
    ndp = np.array(ser[:np.size(dp)])

    for _specie, _m in refractive_dic.items():
        _ndp = ser[_specie] / (1 + ser['ALWC_volume_ratio']) * ndp
        _Ext_dist, _Sca_dist, _Abs_dist = Mie_PESD(_m, wavelength, dp, dlogdp, _ndp)

        ext_dist += _Ext_dist
        sca_dist += _Sca_dist
        abs_dist += _Abs_dist

    return dict(ext=ext_dist, sca=sca_dist, abs=abs_dist)
