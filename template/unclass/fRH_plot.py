from pathlib import Path
from pandas import read_csv
import matplotlib.pyplot as plt
from template import set_figure, unit, getColor

PATH_MAIN = Path('C:/Users/alex/PycharmProjects/DataPlot/Data/Level1')

with open(PATH_MAIN / 'fRH.csv', 'r', encoding='utf-8', errors='ignore') as f:
    frh = read_csv(f).set_index('RH')


@set_figure
def fRH_plot():  # ref f(RH)
    print(f'Plot: fRH_plot')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150, constrained_layout=True)
    plt.plot(frh.index, frh['fRH'], 'k-o', lw=2)
    plt.plot(frh.index, frh['fRHs'], 'g-o', lw=2)
    plt.plot(frh.index, frh['fRHl'], 'r-o', lw=2)
    plt.plot(frh.index, frh['fRHSS'], 'b-o', lw=2)
    plt.xlim(0, 100)
    plt.ylim(1, )
    plt.title(r'$\bf Hygroscopic\ growth\ factor$')
    plt.grid(axis='y', color='gray', linestyle='dashed', linewidth=1, alpha=0.6)
    plt.xlabel(r'$\bf RH\ (\%)$')
    plt.ylabel(r'$\bf f(RH)$')
    plt.legend([fr'$\bf f(RH)_{{original}}$',
                fr'$\bf f(RH)_{{small\ mode}}$',
                fr'$\bf f(RH)_{{large\ mode}}$',
                fr'$\bf f(RH)_{{sea\ salt}}$'],
               loc='upper left', prop=dict(size=16), handlelength=1, frameon=False)
    # fig.savefig('fRH_plot')
    return fig, ax


if __name__ == '__main__':
    fRH_plot()
