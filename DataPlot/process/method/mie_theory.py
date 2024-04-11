import math

import numpy as np
from PyMieScatt import AutoMieQ


def Mie_Q(m: complex,
          wavelength: float,
          dp: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    dp_arr = np.atleast_1d(dp)  # Ensure dp is a numpy array

    result_list = [AutoMieQ(m, wavelength, dp, nMedium=1.0) for dp in dp_arr]
    Q_ext, Q_sca, Q_abs, g, Q_pr, Q_back, Q_ratio = np.array(result_list).T  # Transpose for proper unpacking

    return Q_ext, Q_sca, Q_abs


def Mie_MEE(m: complex,
            wavelength: float,
            dp: float | np.ndarray,
            density: float):
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
    Q_ext, Q_sca, Q_abs = Mie_Q(m, wavelength, dp)

    MEE = (3 * Q_ext) / (2 * density * dp) * 1000
    MSE = (3 * Q_sca) / (2 * density * dp) * 1000
    MAE = (3 * Q_abs) / (2 * density * dp) * 1000

    return MEE, MSE, MAE


def Mie_PESD(m: complex,
             wavelength: float,
             dp: float | np.ndarray,
             ndp: float | np.ndarray):
    """
    Simultaneously calculate "extinction distribution" and "integrated results" using the Mie_Q method.

    Parameters
    ----------
    m : complex
        The complex refractive index of the particles.
    wavelength : float
        The wavelength of the incident light.
    dp : list
        Particle sizes.
    ndp : list
        Number concentration from SMPS or APS in the units of dN/dlogdp.

    Returns
    -------
    result : ...
        return dext/dlogdp. Therefore, please make sure input the dNdlogdp data

    Examples
    --------
    Example usage of the Mie_PESD function:

    >>> m = 1.5 + 0.02j
    >>> wavelength = 550  # in nm
    >>> dp = [0.1, 0.2, 0.5, 1.0]  # list of particle diameters in micrometers
    >>> ndp = [100, 50, 30, 20]  # number concentration of particles
    >>> result = Mie_PESD(m, wavelength, dp, ndp)
    """
    Q_ext, Q_sca, Q_abs = Mie_Q(m, wavelength, dp)

    # dN / dlogdp
    dNdlogdp = np.atleast_1d(ndp)

    # dN = equal to the area under n(dp)
    # The 1E-6 here is so that the final value is the same as 1/10^6m.
    Ext = Q_ext * (math.pi / 4 * dp ** 2) * dNdlogdp * 1e-6
    Sca = Q_sca * (math.pi / 4 * dp ** 2) * dNdlogdp * 1e-6
    Abs = Q_abs * (math.pi / 4 * dp ** 2) * dNdlogdp * 1e-6

    return Ext, Sca, Abs


def Mie_Lognormal(m: complex,
                  wavelength: float,
                  geoMean: float,
                  geoStdDev: float,
                  numberOfParticles: float,
                  numberOfBins: int = 167,
                  lower: float = 1,
                  upper: float = 2500,
                  gamma: float = 1):
    """
    Calculate Mie scattering properties for a lognormal particle size distribution.

    Parameters:
    - m (complex): Complex refractive index of the particle.
    - wavelength (float): Wavelength of the incident light.
    - geoMean (float): Geometric mean of the particle size distribution.
    - geoStdDev (float): Geometric standard deviation of the particle size distribution.
    - numberOfParticles (float): Number of particles.
    - numberOfBins (int, optional): Number of bins for the lognormal distribution. Default is 167.
    - lower (float, optional): Lower limit of the particle size distribution. Default is 1.
    - upper (float, optional): Upper limit of the particle size distribution. Default is 2500.
    - gamma (float, optional): Parameter for lognormal distribution. Default is 1.

    Returns:
    - tuple: A tuple containing the extinction cross-section (Bext), scattering cross-section (Bsca),
      and absorption cross-section (Babs).

    Example:
    >>> m = complex(1.5, 0.02)
    >>> wavelength = 0.55
    >>> geoMean = 300
    >>> geoStdDev = 1.5
    >>> numberOfParticles = 1e6
    >>> Bext, Bsca, Babs = Mie_Lognormal(m, wavelength, geoMean, geoStdDev, numberOfParticles)
    ```

    Note:
    The function uses the Mie_PESD function to calculate the scattering properties based on a lognormal size distribution.
    """
    ithPart = lambda gammai, dp, gmd, gsd: (
            (gammai / (np.sqrt(2 * np.pi) * np.log(gsd) * dp)) *
            np.exp(-(np.log(dp) - np.log(gmd)) ** 2 / (2 * np.log(gsd) ** 2)))

    dp = np.logspace(np.log10(lower), np.log10(upper), numberOfBins)

    ndp = numberOfParticles * ithPart(gamma, dp, geoMean, geoStdDev)

    Bext, Bsca, Babs = Mie_PESD(m, wavelength, dp, ndp)

    return Bext, Bsca, Babs


if __name__ == '__main__':
    m = 1.5 + 0.02j
    wavelength = 550  # in nm
    dp = np.array([0.1, 0.2, 0.5, 1.0, 11, 150] * 100)  # particle diameters in micrometers
    result = Mie_Q(m, wavelength, dp)
