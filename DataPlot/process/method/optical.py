import numpy as np

from .mie_theory import Mie_PESD


def internal(ser, dp, dlogdp, wavelength=550) -> dict:
    m = ser['n_amb'] + 1j * ser['k_amb']
    ndp = np.array(ser[:np.size(dp)])
    ext_dist, sca_dist, abs_dist = Mie_PESD(m, wavelength, dp, ndp)

    return dict(ext=ext_dist, sca=sca_dist, abs=abs_dist)


def external(ser, dp, dlogdp, wavelength=550) -> dict:
    refractive_dic = {'AS_volume_ratio':   1.53 + 0j,
                      'AN_volume_ratio':   1.55 + 0j,
                      'OM_volume_ratio':   1.54 + 0j,
                      'Soil_volume_ratio': 1.56 + 0.01j,
                      'SS_volume_ratio':   1.54 + 0j,
                      'EC_volume_ratio':   1.80 + 0.54j,
                      'ALWC_volume_ratio': 1.333 + 0j}

    ndp = np.array(ser[:np.size(dp)])
    mie_results = (
        Mie_PESD(refractive_dic[_specie], wavelength, dp, ser[_specie] / (1 + ser['ALWC_volume_ratio']) * ndp) for
        _specie in refractive_dic)

    ext_dist, sca_dist, abs_dist = (np.sum([res[0] for res in mie_results], axis=0),
                                    np.sum([res[1] for res in mie_results], axis=0),
                                    np.sum([res[2] for res in mie_results], axis=0))

    return dict(ext=ext_dist, sca=sca_dist, abs=abs_dist)


def fix_PNSD(ser, dp, dlogdp, PNSD, wavelength=550):
    m = ser['n_amb'] + 1j * ser['k_amb']
    ndp = PNSD
    ext_dist, sca_dist, abs_dist = Mie_PESD(m, wavelength, dp, ndp)

    return np.sum(ext_dist * dlogdp)


def fix_RI(ser, dp, dlogdp, RI, wavelength=550):
    m = RI
    ndp = np.array(ser[:np.size(dp)])
    ext_dist, sca_dist, abs_dist = Mie_PESD(m, wavelength, dp, ndp)

    return np.sum(ext_dist * dlogdp)
