from pathlib import Path
from typing import Literal

import numpy as np
from pandas import DataFrame

from DataPlot.process.core import *
from DataPlot.process.script.DistCalculator import (NumberDistCalculator, SurfaceDistCalculator, VolumeDistCalculator,
                                                    PropertiesDistCalculator, ExtinctionDistCalculator)


class SizeDist:
    """
    Attributes
    ----------

    _data: DataFrame
        The processed PSD data stored as a pandas DataFrame.

    _dp: ndarray
        The array of particle diameters from the PSD data.

    _dlogdp: ndarray
        The array of logarithmic particle diameter bin widths.

    _index: DatetimeIndex
        The index of the DataFrame representing time.

    _state: str
        The state of particle size distribution data.

    Methods
    -------
    number()
        Calculate number distribution properties.

    surface(filename='PSSD_dSdlogdp.csv')
        Calculate surface distribution properties.

    volume(filename='PVSD_dVdlogdp.csv')
        Calculate volume distribution properties.

    """

    def __init__(self,
                 filename: Path | str = None,
                 data: DataFrame = None,
                 state: Literal['ddp', 'dlogdp'] = 'dlogdp',
                 weighting: Literal['n', 's', 'v', 'ext_in', 'ext_ex'] = 'n'
                 ):

        self._data: DataFrame = DataReader(filename) if data is None else data
        self._dp = np.array(self._data.columns, dtype=float)
        self._dlogdp = np.full_like(self._dp, 0.014)
        self._index = self._data.index.copy()
        self._state = state
        self._weighting = weighting

    @property
    def data(self) -> DataFrame:
        return self._data

    @property
    def dp(self) -> np.ndarray:
        return self._dp

    @dp.setter
    def dp(self, new_dp: np.ndarray):
        self._dp = new_dp

    @property
    def dlogdp(self) -> np.ndarray:
        return self._dlogdp

    @dlogdp.setter
    def dlogdp(self, new_dlogdp: np.ndarray):
        self._dlogdp = new_dlogdp

    @property
    def index(self):
        return self._index

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if value not in ['dlogdp', 'ddp']:
            raise ValueError("state must be 'dlogdp' or 'ddp'")
        self._state = value

    @property
    def weighting(self):
        return self._weighting

    def number(self) -> DataFrame:
        """ Calculate number distribution """
        return NumberDistCalculator(self).calculate()

    def surface(self, save_filename: Path | str = None) -> DataFrame:
        """ Calculate surface distribution """
        surface_dist = SurfaceDistCalculator(self).calculate()

        if save_filename:
            surface_dist.to_csv(save_filename)

        return surface_dist

    def volume(self, save_filename: Path | str = None) -> DataFrame:
        """ Calculate volume distribution """
        volume_dist = VolumeDistCalculator(self).calculate()

        if save_filename:
            volume_dist.to_csv(save_filename)

        return volume_dist

    def properties(self) -> DataFrame:
        """ Calculate properties of distribution """
        return PropertiesDistCalculator(self).calculate()

    def extinction(self,
                   RI: DataFrame,
                   method: Literal['internal', 'external', 'core-shell', 'sensitivity'],
                   result_type: Literal['extinction', 'scattering', 'absorption'] = 'extinction',
                   save_filename: Path | str = None
                   ) -> DataFrame:
        """ Calculate volume distribution """
        ext_dist = ExtinctionDistCalculator(self, RI, method, result_type).calculate()

        if save_filename:
            ext_dist.to_csv(save_filename)

        return ext_dist
