import matplotlib.pyplot as plt
from DataPlot.process import DataReader
from DataPlot.plot import set_figure


@set_figure
def fRH_plot() -> plt.Axes:
    print('Plot: fRH_plot')
    frh = DataReader('fRH.json')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
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
    return ax


if __name__ == '__main__':
    fRH_plot()
