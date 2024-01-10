from PyMieScatt import AutoMieQ, Mie_SD
import numpy as np
import math
import pandas as pd


def Mie_Q(m, wavelength, dp):
    """
    Calculate Mie scattering efficiency (Q) for a distribution of spherical particles.

    Parameters
    ----------
    m : complex
        The complex refractive index of the particles.
    wavelength : float
        The wavelength of the incident light.
    dp : ndarray
        The array of particle diameters.

    Returns
    -------
    Q : ndarray
        The Mie scattering efficiency for each particle diameter.

    Examples
    --------
    Example usage of the Mie_Q function:

    >>> m = 1.5 + 0.02j
    >>> wavelength = 550  # in nm
    >>> dp = np.array([0.1, 0.2, 0.5, 1.0])  # particle diameters in micrometers
    >>> result = Mie_Q(m, wavelength, dp)
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
    """
    Calculate mass extinction efficiency and other parameters.

    Parameters
    ----------
    m : complex
        The complex refractive index of the particles.
    wavelength : float
        The wavelength of the incident light.
    dp : list or float
        List of particle sizes or a single value.
    density : float
        The density of particles.

    Returns
    -------
    result : ...
        Description of the result.

    Examples
    --------
    Example usage of the Mie_MEE function:

    >>> m = 1.5 + 0.02j
    >>> wavelength = 550  # in nm
    >>> dp = [0.1, 0.2, 0.5, 1.0]  # list of particle diameters in micrometers
    >>> density = 1.2  # density of particles
    >>> result = Mie_MEE(m, wavelength, dp, density)
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
    """ Mie_SD
    Simultaneously calculate "extinction distribution" and "integrated results" using the Mie_Q method.

    Parameters
    ----------
    m : complex
        The complex refractive index of the particles.
    wavelength : float
        The wavelength of the incident light.
    dp : list
        Particle sizes.
    dlogdp : list
        Logarithmic particle diameter bin widths.
    ndp : list
        Number concentration from SMPS or APS in the units of dN/dlogdp.

    Returns
    -------
    result : ...
        Description of the result.

    Examples
    --------
    Example usage of the Mie_PESD function:

    >>> m = 1.5 + 0.02j
    >>> wavelength = 550  # in nm
    >>> dp = [0.1, 0.2, 0.5, 1.0]  # list of particle diameters in micrometers
    >>> dlogdp = [0.05, 0.1, 0.2, 0.5]  # list of dlogdp
    >>> ndp = [100, 50, 30, 20]  # number concentration of particles
    >>> result = Mie_PESD(m, wavelength, dp, dlogdp, ndp)
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


def mie_theory(df_psd, df_m, wave_length=550):
    _ori_idx = df_psd.index.copy()
    _cal_idx = df_psd.loc[df_m.dropna().index].dropna(how='all').index

    _psd, _RI = df_psd.loc[_cal_idx], df_m.loc[_cal_idx]

    ## parameter
    _bins = _psd.keys().tolist()

    ## calculate
    _dt_lst = []
    for _dt, _m in zip(_psd.values, _RI.values):
        _out_dic = Mie_SD(_m, wave_length, _bins, _dt, asDict=True)
        _dt_lst.append(_out_dic)

    _out = pd.DataFrame(_dt_lst, index=_cal_idx).reindex(_ori_idx)

    if len(_out.dropna()) == 0:
        return _out

    _out = _out.rename(columns={'Bext': 'ext',
                                'Bsca': 'sca',
                                'Babs': 'abs',
                                'Bback': 'back',
                                'Bratio': 'ratio',
                                'Bpr'	: 'pr' ,})

    return _out[['abs', 'sca', 'ext', 'back', 'ratio', 'pr', 'G']]


