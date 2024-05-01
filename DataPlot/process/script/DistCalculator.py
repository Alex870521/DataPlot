from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from pandas import DataFrame, concat

from DataPlot.process.method import properties, internal, external, core_shell, sensitivity


class DistCalculator(ABC):
    @abstractmethod
    def calculate(self) -> DataFrame:
        pass


class NumberDistCalculator(DistCalculator):
    def __init__(self, psd):
        self.psd = psd

    def calculate(self) -> DataFrame:
        return self.psd.data


class SurfaceDistCalculator(DistCalculator):
    def __init__(self, psd):
        self.psd = psd

    def calculate(self) -> DataFrame:
        return self.psd.data.dropna().apply(lambda col: np.pi * self.psd.dp ** 2 * np.array(col),
                                            axis=1, result_type='broadcast').reindex(self.psd.index)


class VolumeDistCalculator(DistCalculator):
    def __init__(self, psd):
        self.psd = psd

    def calculate(self) -> DataFrame:
        return self.psd.data.dropna().apply(lambda col: np.pi / 6 * self.psd.dp ** 3 * np.array(col),
                                            axis=1, result_type='broadcast').reindex(self.psd.index)


class PropertiesDistCalculator(DistCalculator):

    def __init__(self, psd):
        self.psd = psd

    def calculate(self):
        return self.psd.data.dropna().apply(partial(properties, dp=self.psd.dp, dlogdp=self.psd.dlogdp,
                                                    weighting=self.psd.weighting),
                                            axis=1, result_type='expand').reindex(self.psd.index)


# TODO:
class ExtinctionDistCalculator(DistCalculator):
    mapping = {'internal': internal,
               'external': external,
               'core_shell': core_shell,
               'sensitivity': sensitivity}

    def __init__(self, psd, RI, method, result_type):
        self.psd = psd
        self.RI = RI
        self.combined_data = concat([self.psd.data, self.RI], axis=1).dropna()

        self.method = ExtinctionDistCalculator.mapping[method]
        self.result_type = result_type

    def calculate(self) -> DataFrame:
        return self.combined_data.apply(partial(self.method, dp=self.psd.dp, result_type=self.result_type),
                                        axis=1, result_type='expand').reindex(self.psd.index).set_axis(self.psd.dp,
                                                                                                       axis=1)


# TODO:
class LungDepositsDistCalculator(DistCalculator):

    def __init__(self, psd, lung_curve):
        self.psd = psd
        self.lung_curve = lung_curve

    def calculate(self) -> DataFrame:
        pass
