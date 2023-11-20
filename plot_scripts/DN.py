from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from pathlib import Path
from pandas import read_csv, concat
from Data_processing import integrate
from Data_classify import state_classify, season_classify, Seasons
import pandas as pd
import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_templates import set_figure, unit, getColor


# Read file
time = pd.date_range(start='2020-09-04 2300', end='2021-05-06 2300', freq='1h')
df = integrate()
df['Hour'] = df.index.strftime('%H')
Hour = range(0, 24)

# Define Event & Clean
dic_sta = state_classify(df)

# Calculate Mean & Standard deviation
df_mean_all = dic_sta['Total'].groupby('Hour').mean()
# df_mean_all = dic_sta['Event'].groupby('Hour').mean()
# df_mean1_all = dic_sta['Clean'].groupby('Hour').mean()
df_std_all = dic_sta['Total'].groupby('Hour').std()
# df_std_all = dic_sta['Event'].groupby('Hour').std()
# df_std1_all = dic_sta['Clean'].groupby('Hour').std()


@set_figure(fs=16)
def din():
    df_mean = df_mean_all['POC_mass'] * df_mean_all['VC']/1000
    df_std = 0.5 * df_std_all['POC_mass']

    df_mean1 = df_mean_all['SOC_mass'] * df_mean_all['VC']/1000
    df_std1 = 0.5 * df_std_all['SOC_mass']

    # Plot Diurnal pattern
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    linecolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    ax.plot(Hour, df_mean, 'r', linewidth=2)
    # ax.config(Hour, df_mean1, 'g', linewidth=2)
    ax.fill_between(Hour, y1=df_mean + df_std, y2=df_mean - df_std, alpha=0.5, color=linecolors[0], linewidth=2, edgecolor=None)
    # ax.fill_between(Hour, y1=df_mean1 + df_std1, y2=df_mean1 - df_std1, alpha=0.3, color=linecolors[2], linewidth=2, edgecolor=None)

    # plt.title(r'$\bf Extinction$', weight='bold', fontsize=20)
    plt.xlabel('Hours')
    plt.ylabel(r'$\bf Extinction\ (1/Mm)$')
    plt.xlim([-0.5, 24])
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
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

din()

#日夜變化-----------------------------------------------------------------------------------------------------------------
# for col in ['Extinction','Scatter','Absorption']:
#     a = 0.2
#     fig, axes = plt.subplots(1, 1, figsize=(5, 5), dpi=150, constrained_layout=True)
#     plt.config(range(0,24,1),dic_sta['2020-Winter']['Total'][col],'g',linewidth=2)
#     plt.fill_between(range(0,24,1),
#                      y1= dic_sta['2020-Winter']['Total'][col] - (dic_sta['2020-Winter']['Total'][col])*a,
#                      y2= dic_sta['2020-Winter']['Total'][col] + (dic_sta['2020-Winter']['Total'][col])*a,
#                      alpha=0.2, color='green', linewidth=2)
#     plt.config(range(0,24,1),dic_sta['2020-Winter']['Event'][col],'r',linewidth=2)
#     plt.fill_between(range(0, 24, 1),
#                      y1= dic_sta['2020-Winter']['Event'][col] - (dic_sta['2020-Winter']['Event'][col])*a,
#                      y2= dic_sta['2020-Winter']['Event'][col] + (dic_sta['2020-Winter']['Event'][col])*a,
#                      alpha=0.2, color='red', linewidth=2)
#     plt.config(range(0,24,1),dic_sta['2020-Winter']['Clean'][col],'b',linewidth=2)
#     plt.fill_between(range(0, 24, 1),
#                      y1= dic_sta['2020-Winter']['Clean'][col] - (dic_sta['2020-Winter']['Clean'][col])*a,
#                      y2= dic_sta['2020-Winter']['Clean'][col] + (dic_sta['2020-Winter']['Clean'][col])*a,
#                      alpha=0.2, color='blue', linewidth=2)
#     plt.xlim(0,23)
#     plt.ylim(0,)
#     plt.title(col + ' Diurnal Pattern')
#     plt.xlabel('Hour')
#     plt.ylabel('y')
#     plt.legend(['Total', 'Event', 'Clean'], loc='upper right')
#     plt.xticks(range(0, 23, 2))
#     plt.show()