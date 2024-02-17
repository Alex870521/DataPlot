from matplotlib.ticker import (AutoMinorLocator)
import matplotlib.pyplot as plt
from DataPlot.plot import set_figure, unit
from DataPlot.process import data
from DataPlot.plot.scripts import StateClassifier

# Read file
df = data
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
    # process data
    df_mean = data[f'{y}'] * data['VC'] / 1000
    df_std = df_std_all['POC'] * std_area

    # Plot Diurnal pattern
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    linecolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    ax.plot(Hour, df_mean, 'r', linewidth=2)

    ax.fill_between(Hour, y1=df_mean + df_std, y2=df_mean - df_std, alpha=0.5, color=linecolors[0], linewidth=2,
                    edgecolor=None)
    # ax.fill_between(Hour, y1=df_mean1 + df_std1, y2=df_mean1 - df_std1, alpha=0.3, color=linecolors[2], linewidth=2, edgecolor=None)

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
    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText = True)

    plt.show()


if __name__ == '__main__':
    diurnal(df_mean_all, y='POC', std_area=0.5)
