from matplotlib.ticker import (AutoMinorLocator)
import matplotlib.pyplot as plt
from DataPlot.plot import set_figure, unit
from DataPlot.process import *
from DataPlot.plot import *

# Read file
df = DataBase

# Define Event & Clean
df_mean_all, df_std_all = Classifier(df, 'Hour', statistic='Table')


@set_figure(figsize=(6, 6), fs=16)
def diurnal(data, data2, y, ax=None, std_area=0.5):
    print(f'Plot: diurnal of {y}')

    Hour = range(0, 24)

    if ax is None:
        fig, ax = plt.subplots()

    df_mean = data[f'{y}']
    df_std = data2[f'{y}'] * std_area

    # Plot Diurnal pattern
    linecolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    ax.plot(Hour, df_mean, 'r', linewidth=2)

    ax.fill_between(Hour, y1=df_mean + df_std, y2=df_mean - df_std, alpha=0.5, color=linecolors[0], linewidth=2,
                    edgecolor=None)

    # plt.title(r'$\bf Extinction$', weight='bold', fontsize=20)
    plt.xlabel('Hours')
    plt.ylabel(unit(f'{y}'))
    plt.xlim([0, 23])
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    plt.ylim(0, 5)
    # plt.legend(['Event','Clean','Event_SD','Clean_SD'], prop = prop_legend, loc='upper left', frameon=False)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='x', which='minor')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    plt.show()

    return ax


if __name__ == '__main__':
    diurnal(df_mean_all, df_std_all, y='SOC', std_area=0.5)
