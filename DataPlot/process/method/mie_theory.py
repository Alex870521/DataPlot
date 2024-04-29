import math
from typing import Sequence

import numpy as np
from PyMieScatt import AutoMieQ
from numpy import exp, log, log10, sqrt, pi


def Mie_Q(m: complex,
          wavelength: float,
          dp: float | Sequence[float]
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Mie scattering efficiency (Q) for given spherical particle diameter(s).

    Parameters
    ----------
    m : complex
        The complex refractive index of the particles.
    wavelength : float
        The wavelength of the incident light (in nm).
    dp : float | Sequence[float]
        Particle diameters (in nm), can be a single value or Sequence object.

    Returns
    -------
    Q_ext : ndarray
        The Mie extinction efficiency for each particle diameter.
    Q_sca : ndarray
        The Mie scattering efficiency for each particle diameter.
    Q_abs : ndarray
        The Mie absorption efficiency for each particle diameter.

    Examples
    --------
    >>> Q_ext, Q_sca, Q_abs = Mie_Q(m=complex(1.5, 0.02), wavelength=550, dp=[100, 200, 300, 400])
    """
    # Ensure dp is a numpy array
    dp = np.atleast_1d(dp)

    # Transpose for proper unpacking
    Q_ext, Q_sca, Q_abs, g, Q_pr, Q_back, Q_ratio = np.array([AutoMieQ(m, wavelength, _dp) for _dp in dp]).T

    return Q_ext, Q_sca, Q_abs


def Mie_MEE(m: complex,
            wavelength: float,
            dp: float | Sequence[float],
            density: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate mass extinction efficiency and other parameters.

    Parameters
    ----------
    m : complex
        The complex refractive index of the particles.
    wavelength : float
        The wavelength of the incident light.
    dp : float | Sequence[float]
        List of particle sizes or a single value.
    density : float
        The density of particles.

    Returns
    -------
    MEE : ndarray
        The mass extinction efficiency for each particle diameter.
    MSE : ndarray
        The mass scattering efficiency for each particle diameter.
    MAE : ndarray
        The mass absorption efficiency for each particle diameter.

    Examples
    --------
    >>> MEE, MSE, MAE = Mie_MEE(m=complex(1.5, 0.02), wavelength=550, dp=[100, 200, 300, 400], density=1.2)
    """
    Q_ext, Q_sca, Q_abs = Mie_Q(m, wavelength, dp)

    MEE = (3 * Q_ext) / (2 * density * dp) * 1000
    MSE = (3 * Q_sca) / (2 * density * dp) * 1000
    MAE = (3 * Q_abs) / (2 * density * dp) * 1000

    return MEE, MSE, MAE


def Mie_PESD(m: complex,
             wavelength: float = 550,
             dp: float | Sequence[float] = None,
             ndp: float | Sequence[float] = None,
             lognormal: bool = False,
             dp_range: tuple = (1, 2500),
             geoMean: float = 200,
             geoStdDev: float = 2,
             numberOfParticles: float = 1e6,
             numberOfBins: int = 167,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simultaneously calculate "extinction distribution" and "integrated results" using the Mie_Q method.

    Parameters
    ----------
    m : complex
        The complex refractive index of the particles.
    wavelength : float
        The wavelength of the incident light.
    dp : float | Sequence[float]
        Particle sizes.
    ndp : float | Sequence[float]
        Number concentration from SMPS or APS in the units of dN/dlogdp.
    lognormal : bool, optional
        Whether to use lognormal distribution for ndp. Default is False.
    dp_range : tuple, optional
        Range of particle sizes. Default is (1, 2500) nm.
    geoMean : float, optional
        Geometric mean of the particle size distribution. Default is 200 nm.
    geoStdDev : float, optional
        Geometric standard deviation of the particle size distribution. Default is 2.
    numberOfParticles : float, optional
        Number of particles. Default is 1e6.
    numberOfBins : int, optional
        Number of bins for the lognormal distribution. Default is 167.

    Returns
    -------
    ext_dist : ndarray
        The extinction distribution for the given data.
    sca_dist : ndarray
        The scattering distribution for the given data.
    abs_dist : ndarray
        The absorption distribution for the given data.

    Notes
    -----
    return in "dext/dlogdp", please make sure input the dNdlogdp data.

    Examples
    --------
    >>> Ext, Sca, Abs = Mie_PESD(m=complex(1.5, 0.02), wavelength=550, dp=[100, 200, 500, 1000], ndp=[100, 50, 30, 20])
    """
    if lognormal:
        dp = np.logspace(log10(dp_range[0]), log10(dp_range[1]), numberOfBins)

        ndp = numberOfParticles * (1 / (log(geoStdDev) * sqrt(2 * pi)) *
                                   exp(-(log(dp) - log(geoMean)) ** 2 / (2 * log(geoStdDev) ** 2)))

    # dN / dlogdp
    ndp = np.atleast_1d(ndp)

    Q_ext, Q_sca, Q_abs = Mie_Q(m, wavelength, dp)

    # The 1e-6 here is so that the final value is the same as the unit 1/10^6m.
    Ext = Q_ext * (math.pi / 4 * dp ** 2) * ndp * 1e-6
    Sca = Q_sca * (math.pi / 4 * dp ** 2) * ndp * 1e-6
    Abs = Q_abs * (math.pi / 4 * dp ** 2) * ndp * 1e-6

    return Ext, Sca, Abs


if __name__ == '__main__':
    result = Mie_Q(m=complex(1.5, 0.02), wavelength=550, dp=[100., 200.])
