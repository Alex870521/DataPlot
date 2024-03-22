import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot import *
from scipy.stats import pearsonr

df = DataBase


if __name__ == '__main__':
    data = DataBase
    plot.corr_matrix(data)
