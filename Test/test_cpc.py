import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.plot_templates import set_figure, unit
from pathlib import Path

PATH_MAIN = Path(__file__).parent

df1 = (pd.read_csv(PATH_MAIN / '20231120-CPC1.csv', usecols=range(7), header=17, skipfooter=8, engine='python')
       .set_index('Elapsed Time'))
df2 = (pd.read_csv(PATH_MAIN / '20231120-CPC2.csv', usecols=range(7), header=17, skipfooter=8, engine='python')
       .set_index('Elapsed Time'))

new_column_names = ['60 nm', '60* nm', '60** nm', '80 nm', '100 nm', '150 nm']
df1.columns = df2.columns = new_column_names
dic = {f'{name}': pd.concat([df1[f'{name}'], df2[f'{name}']], axis=1) for name in new_column_names}


@set_figure
def cpc_vs():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    x, xerr = [], []
    y, yerr = [], []
    for _label, _df in dic.items():
        _df.columns = ['CPC_1', 'CPC_2']
        x.append(_df['CPC_1'].mean())
        xerr.append(_df['CPC_1'].std())
        y.append(_df['CPC_2'].mean())
        yerr.append(_df['CPC_2'].std())
        plt.scatter(_df['CPC_1'], _df['CPC_2'], label=f'{_label}')

    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=5, label='_nolegend_')
    xlim = [df1['150 nm'].min() * 0.8, df1['60 nm'].max() * 1.05]
    ylim = [df2['150 nm'].min() * 0.8, df2['60 nm'].max() * 1.05]
    xlabel = unit('CPC_1')
    ylabel = unit('CPC_2')
    ax.legend(loc='best')
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 3), useMathText=True)

    axins = inset_axes(ax, width="30%", height="30%", loc='upper center')
    axins.scatter(dic['60 nm']['CPC_1'], dic['60 nm']['CPC_2'], label='Zoomed In')
    axins.scatter(dic['60* nm']['CPC_1'], dic['60* nm']['CPC_2'], label='Zoomed In')
    axins.scatter(dic['60** nm']['CPC_1'], dic['60** nm']['CPC_2'], label='Zoomed In')
    # 设置放大图坐标轴的范围
    x1, x2, y1, y2 = 2.22e5, 2.33e5, 2.51e5, 2.64e5  # 设置放大区域的范围
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.ticklabel_format(axis='both', style='sci', scilimits=(0, 3), useMathText=True)

    # 将放大图坐标轴的刻度去掉
    axins.set_xticks([])
    axins.set_yticks([])

    # 添加连接线
    ax.indicate_inset_zoom(axins, edgecolor="black")


if __name__ == '__main__':
    cpc_vs()
