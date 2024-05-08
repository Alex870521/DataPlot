from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from pandas import DataFrame, concat

from DataPlot.process.method import properties, internal, external, core_shell, sensitivity


class AbstractDistCalc(ABC):
    @abstractmethod
    def useApply(self) -> DataFrame:
        pass


class NumberDistCalc(AbstractDistCalc):
    def __init__(self, psd):
        self.psd = psd

    def useApply(self) -> DataFrame:
        return self.psd.data


class SurfaceDistCalc(AbstractDistCalc):
    def __init__(self, psd):
        self.psd = psd

    def useApply(self) -> DataFrame:
        return self.psd.data.dropna().apply(lambda col: np.pi * self.psd.dp ** 2 * np.array(col),
                                            axis=1, result_type='broadcast').reindex(self.psd.index)


class VolumeDistCalc(AbstractDistCalc):
    def __init__(self, psd):
        self.psd = psd

    def useApply(self) -> DataFrame:
        return self.psd.data.dropna().apply(lambda col: np.pi / 6 * self.psd.dp ** 3 * np.array(col),
                                            axis=1, result_type='broadcast').reindex(self.psd.index)


class PropertiesDistCalc(AbstractDistCalc):

    def __init__(self, psd):
        self.psd = psd

    def useApply(self):
        return self.psd.data.dropna().apply(partial(properties, dp=self.psd.dp, dlogdp=self.psd.dlogdp,
                                                    weighting=self.psd.weighting),
                                            axis=1, result_type='expand').reindex(self.psd.index)


class ExtinctionDistCalc(AbstractDistCalc):
    mapping = {'internal': internal,
               'external': external,
               'core_shell': core_shell,
               'sensitivity': sensitivity}

    def __init__(self, psd, RI, method, result_type):
        self.psd = psd
        self.RI = RI
        self.combined_data = concat([self.psd.data, self.RI], axis=1).dropna()

        self.method = ExtinctionDistCalc.mapping[method]
        self.result_type = result_type

    def useApply(self) -> DataFrame:
        return self.combined_data.apply(partial(self.method, dp=self.psd.dp, result_type=self.result_type),
                                        axis=1, result_type='expand').reindex(self.psd.index).set_axis(self.psd.dp,
                                                                                                       axis=1)


# TODO:
class LungDepositsDistCalc(AbstractDistCalc):

    def __init__(self, psd, lung_curve):
        self.psd = psd
        self.lung_curve = lung_curve

    def useApply(self) -> DataFrame:
        pass


class CalculatorInterface:
    def __init__(self, calculator: AbstractDistCalc):
        self.calculator = calculator

    def useApply(self) -> DataFrame:
        return self.calculator.useApply()
