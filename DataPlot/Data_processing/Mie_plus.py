from PyMieScatt.Mie import AutoMieQ
import numpy as np
import math


def Mie_Q(m, wavelength, dp):
    """  Single-particle extinction efficiency and others parameter

    :param m: refractive index
    :param wavelength: incident wavelength
    :param dp: list of particle sizes or a single value
    :return:
    """
    nMedium = 1.0
    m /= nMedium
    wavelength /= nMedium
    try:
        _length = len(dp)
        result_list = list(map(lambda i: AutoMieQ(m, wavelength, dp[i], nMedium), range(_length)))
        Q_ext, Q_sca, Q_abs, g, Q_pr, Q_back, Q_ratio = map(np.array, zip(*result_list))

    except TypeError:
        _length = 1
        Q_ext, Q_sca, Q_abs, g, Q_pr, Q_back, Q_ratio = AutoMieQ(m, wavelength, dp, nMedium)

    return Q_ext, Q_sca, Q_abs


def Mie_MEE(m, wavelength, dp, density):
    """ Mass extinction efficiency and others parameter

    :param m: refractive index
    :param wavelength: incident wavelength
    :param dp: list of particle sizes or a single value
    :param density: density
    :return:
    """
    nMedium = 1.0
    m /= nMedium
    wavelength /= nMedium
    _length = len(dp)

    result_list = list(map(lambda i: AutoMieQ(m, wavelength, dp[i], nMedium), range(_length)))
    Q_ext, Q_sca, Q_abs, g, Q_pr, Q_back, Q_ratio = map(np.array, zip(*result_list))

    MEE = (3 * Q_ext) / (2 * density * dp) * 1000
    MSE = (3 * Q_sca) / (2 * density * dp) * 1000
    MAE = (3 * Q_abs) / (2 * density * dp) * 1000

    return MEE, MSE, MAE


def Mie_PESD(m, wavelength, dp, dlogdp, ndp):
    """ Simultaneous calculation of "extinction distribution" and "integrated results" using the --Mie_Q-- method

    :param m: refractive index
    :param wavelength: incident wavelength
    :param dp: list of particle sizes
    :param dlogdp: list of dlogdp
    :param ndp: number concentration from SMPS or APS in the units of dN/dlogdp
    :param output_dist:
    :return:
    """

    # Q
    Q_ext, Q_sca, Q_abs = Mie_Q(m, wavelength, dp)

    # dN / dlogdp
    dNdlogdp = ndp

    # dN = equal to the area under n(dp)
    # The 1E-6 here is so that the final value is the same as 1/10^6m.
    # return dext/dlogdp
    Ext = Q_ext * (math.pi / 4 * dp ** 2) * dNdlogdp * 1e-6
    Sca = Q_sca * (math.pi / 4 * dp ** 2) * dNdlogdp * 1e-6
    Abs = Q_abs * (math.pi / 4 * dp ** 2) * dNdlogdp * 1e-6

    return Ext, Sca, Abs
