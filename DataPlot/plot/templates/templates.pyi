from matplotlib.axes import Axes
from pandas import DataFrame
from typing import Literal, Union

class Pie:
    @staticmethod
    def pieplot(data_set: Union[DataFrame, dict],
                labels: list[str],
                unit: str,
                style: Literal["pie", 'donut'],
                ax: Axes | None = None,
                symbol: bool = True,
                **kwargs) -> Axes: ...

    @staticmethod
    def donuts(data_set: Union[DataFrame, dict],
               labels: list[str],
               unit: str,
               ax: Axes | None = None,
               symbol: bool = True,
               **kwargs) -> Axes: ...

class Bar:
    @staticmethod
    def barplot(data_set: Union[DataFrame, dict],
                data_std: Union[DataFrame, None],
                labels: list[str],
                unit: str,
                display: Literal["stacked", "dispersed"] = "dispersed",
                orientation: Literal["va", "ha"] = 'va',
                ax: Axes | None = None,
                symbol: bool = True,
                **kwargs) -> Axes: ...

class Violin:
    @staticmethod
    def violin(data_set: Union[DataFrame, dict],
               unit: str,
               ax: Axes | None = None,
               **kwargs) -> Axes: ...
