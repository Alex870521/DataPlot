from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from pathlib import Path
from pandas import read_csv, concat
import pandas as pd
import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot import set_figure, unit, getColor
from DataPlot.data_processing import main
from DataPlot.scripts import StateClassifier

# Read file
df = main()
df['Hour'] = df.index.strftime('%H')
Hour = range(0, 24)

# Define Event & Clean
dic_sta = StateClassifier(df)

# Calculate Mean & Standard deviation
df_mean_all = dic_sta['Total'].groupby('Hour').mean(numeric_only=True)
# df_mean_all = dic_sta['Event'].groupby('Hour').mean(numeric_only=True)
# df_mean1_all = dic_sta['Clean'].groupby('Hour').mean(numeric_only=True)
df_std_all = dic_sta['Total'].groupby('Hour').std(numeric_only=True)
# df_std_all = dic_sta['Event'].groupby('Hour').std()
# df_std1_all = dic_sta['Clean'].groupby('Hour').std()


@set_figure(fs=16)
def diurnal(data, y, ax=None, std_area=0.5):
    df_mean = data['POC_mass'] * data['VC'] / 1000
    df_std = std_area * df_std_all['POC_mass']

    df_mean1 = data['SOC_mass'] * data['VC'] / 1000
    df_std1 = std_area * df_std_all['SOC_mass']

    # Plot Diurnal pattern
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    linecolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    ax.plot(Hour, df_mean, 'r', linewidth=2)

    ax.fill_between(Hour, y1=df_mean + df_std, y2=df_mean - df_std, alpha=0.5, color=linecolors[0], linewidth=2, edgecolor=None)
    # ax.fill_between(Hour, y1=df_mean1 + df_std1, y2=df_mean1 - df_std1, alpha=0.3, color=linecolors[2], linewidth=2, edgecolor=None)

    # plt.title(r'$\bf Extinction$', weight='bold', fontsize=20)
    plt.xlabel('Hours')
    plt.ylabel(r'$\bf Extinction\ (1/Mm)$')
    plt.xlim([0, 23])
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    plt.ylim(0, 5)
    # plt.yticks([0, 50, 100, 150])
    # plt.legend(['Event','Clean','Event_SD','Clean_SD'], prop = prop_legend, loc='upper left', frameon=False)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='x', which='minor')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText = True)

    ax2 = ax.twinx()
    ax2.plot(Hour, df_mean1, 'b', linewidth=2)
    ax2.fill_between(Hour, y1=df_mean1 + df_std1, y2=df_mean1 - df_std1, alpha=0.5, color=linecolors[1], linewidth=2,
                    edgecolor=None)
    ax2.set_ylabel(r'$\bf VC\ (m^2/s)$')
    ax2.set_ylim(0, 5)
    # plt.savefig(Path('Master thesis') / 'Diel' / f"Diel-{key}", transparent=True, bbox_inches="tight")
    plt.show()


diurnal(df_mean_all, y='POC_mass')
