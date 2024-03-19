import matplotlib.pyplot as plt
from DataPlot.process import DataReader
from DataPlot.plot import set_figure
from scipy.optimize import curve_fit


frh = DataReader('fRH.json')


@set_figure
def fRH_plot() -> plt.Axes:
    print('Plot: fRH_plot')
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


@set_figure
def fit_fRH():
    split = [36, 75]
    x1 = frh.index.to_numpy()[split[0]:split[1]]
    y1 = frh['fRHs'].values[split[0]:split[1]]
    x2 = frh.index.to_numpy()[split[1]:]
    y2 = frh['fRHs'].values[split[1]:]

    def f__RH(RH, a, b, c):
        f = a + b * (RH/100)**c
        return f

    popt, pcov = curve_fit(f__RH, x1, y1)
    a, b, c = popt
    yvals = f__RH(x1, a, b, c) #擬合y值
    print(u'係數a:', a)
    print(u'係數b:', b)
    print(u'係數c:', c)

    popt2, pcov2 = curve_fit(f__RH, x2, y2)
    a2, b2, c2 = popt2
    yvals2 = f__RH(x2, a2, b2, c2) #擬合y值
    print(u'係數a2:', a2)
    print(u'係數b2:', b2)
    print(u'係數c2:', c2)

    fig, axes = plt.subplots(1, 1, figsize=(5, 5), dpi=150, constrained_layout=True)
    plt.scatter(x1, y1, label='original values')
    plt.plot(x1, yvals, 'r', label='polyfit values')
    plt.scatter(x2, y2, label='original values')
    plt.plot(x2, yvals2, 'r', label='polyfit values')
    plt.xlabel(r'$\bf RH$')
    plt.ylabel(r'$\bf f(RH)$')
    plt.legend(loc='best')
    plt.title(r'$\bf Curve fit$')
    plt.show()


if __name__ == '__main__':
    # fRH_plot()
    fit_fRH()
