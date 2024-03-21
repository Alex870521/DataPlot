from .core import *
from .templates import *
from .distribution import *
from .improve import *
from .optical import *
from .meteorology import *
from .scripts import time_series, rsm

__all__ = ['set_figure',
           'Unit',
           'Color',
           'ammonium_rich',
           'fRH_plot',
           'wind_rose',
           'wind_heatmap',
           'time_series',
           'rsm',

           'scatter',
           'linear_regression',
           'multiple_linear_regression',

           'Pie',
           'Violin',
           'Bar',
           ]
