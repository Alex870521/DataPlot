from PyMieScatt.Mie import AutoMieQ
import numpy as np
import math


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
    Example usage of the mass_extinction_efficiency function:

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
    """
    Simultaneously calculate "extinction distribution" and "integrated results" using the MIE_Q method.

    Parameters
    ----------
    m : complex
        The complex refractive index of the particles.
    wavelength : float
        The wavelength of the incident light.
    dp : list
        List of particle sizes.
    dlogdp : list
        List of logarithmic particle diameter bin widths.
    ndp : list
        Number concentration from SMPS or APS in the units of dN/dlogdp.
    output_dist : str or None, optional
        Output distribution type ('extinction', 'scattering', 'absorption', 'all', None).
        Defaults to None.

    Returns
    -------
    result : ...
        Description of the result.

    Examples
    --------
    Example usage of the calculate_extinction_distribution function:

    >>> m = 1.5 + 0.02j
    >>> wavelength = 550  # in nm
    >>> dp = [0.1, 0.2, 0.5, 1.0]  # list of particle diameters in micrometers
    >>> dlogdp = [0.05, 0.1, 0.2, 0.5]  # list of dlogdp
    >>> ndp = [100, 50, 30, 20]  # number concentration of particles
    >>> output_dist = 'all'
    >>> result = Mie_PESD(m, wavelength, dp, dlogdp, ndp, output_dist)
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
