import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataPlot.plot_templates import set_figure


def get_group_avgdist_stddist(group):
    avg_line = {}
    std_line = {}
    for name, subdf in group.__iter__():
        avg_line[name] = np.array(subdf.mean()[1:])
        std_line[name] = np.array(subdf.std()[1:])
    return avg_line, std_line


Ext_amb_dis, Ext_amb_dis_std = get_group_avgdist_stddist(Ext_amb_df.dropna().groupby('State'))
Ext_dry_dis, Ext_dry_dis_std = get_group_avgdist_stddist(Ext_dry_df.dropna().groupby('State'))
Ext_amb_dis_external, Ext_amb_dis_std__external = get_group_avgdist_stddist(Ext_amb_df_external.dropna().groupby('State'))

PSD_amb_dis, PSD_amb_dis_std = get_group_avgdist_stddist(PSD_amb_df.dropna().groupby('State'))
PSSD_amb_dis, PSSD_amb_dis_std = get_group_avgdist_stddist(PSSD_amb_df.dropna().groupby('State'))
PVSD_amb_dis, PVSD_amb_dis_std = get_group_avgdist_stddist(PVSD_amb_df.dropna().groupby('State'))


# print('Number')
# print(dist_prop(PSD_amb_dis['Clean']))
# print(dist_prop(PSD_amb_dis['Transition']))
# print(dist_prop(PSD_amb_dis['Event']))
# print('Surface')
# print(dist_prop(PSSD_amb_dis['Clean']))
# print(dist_prop(PSSD_amb_dis['Transition']))
# print(dist_prop(PSSD_amb_dis['Event']))
print('Ext')
print(dist_prop(Ext_amb_dis['Clean']))
print(dist_prop(Ext_amb_dis['Transition']))
print(dist_prop(Ext_amb_dis['Event']))
print(dist_prop(Ext_dry_dis['Clean']))
print(dist_prop(Ext_dry_dis['Transition']))
print(dist_prop(Ext_dry_dis['Event']))


# print('Vol')
# print(dist_prop(PVSD_amb_dis['Clean']))
# print(dist_prop(PVSD_amb_dis['Transition']))
# print(dist_prop(PVSD_amb_dis['Event']))


def fitting(dp, dist, cut):
    import numpy as np
    from scipy.stats import lognorm
    from scipy.optimize import curve_fit

    # 假設您的兩個峰值資訊為 peak1 和 peak2
    peak, _ = find_peaks(np.concatenate(([min(dist)], dist, [min(dist)])), distance=6)

    peak1 = dp[peak[0]-1]
    peak2 = dp[peak[1]-1]

    Num = np.sum(dist * dlogdp[:-cut])
    data = dist / Num

    # 定義兩個對數常態分佈的函數
    def lognorm_func(dp, N1, mu1, sigma1, N2, mu2, sigma2):
        return (N1 / (np.log(sigma1) * np.sqrt(2 * np.pi)) * np.exp(-(np.log(dp) - np.log(mu1)) ** 2 / (2 * np.log(sigma1) ** 2)) +
                N2 / (np.log(sigma2) * np.sqrt(2 * np.pi)) * np.exp(-(np.log(dp) - np.log(mu2)) ** 2 / (2 * np.log(sigma2) ** 2)))

    # 使用 curve_fit 函數進行擬合
    x = np.arange(1, len(data) + 1)
    popt, pcov = curve_fit(lognorm_func, dp, data, bounds=(0, 1000))

    # 獲取擬合的參數
    N1, mu1, sigma1_fit, N2, mu2, sigma2_fit = popt

    print("擬合結果:")
    print("第一個對數常態分佈:")
    print("Number 1:", N1)
    print("平均值 (mu1):", mu1)
    print("標準差 (sigma1):", sigma1_fit)
    print()
    print("第二個對數常態分佈:")
    print("Number 2:", N2)
    print("平均值 (mu2):", mu2)
    print("標準差 (sigma2):", sigma2_fit)

    plt.plot(dp, data, 'bo', label='Data')  # 原始數據
    plt.plot(dp, lognorm_func(dp, N1, mu1, sigma1_fit, N2, mu2, sigma2_fit), 'r-', label='Fitted Curve')  # 擬合曲線
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.semilogx()
    plt.legend()
    plt.show()


# fitting(dp[:-50], PSSD_amb_dis['Clean'][:-50], cut=30)
# fitting(dp[:-1], Ext_amb_dis['Clean'][:-1], cut=1)

table = pd.pivot_table(df, values=df.keys()[:-5], index='State', columns='Diurnal',
                       aggfunc=np.mean)


color_choose = {'Clean': ['#1d4a9f', '#84a7e9'],
                'Transition': ['#4a9f1d', '#a7e984'],
                'Event': ['#9f1d4a', '#e984a7']}


@set_figure(figsize=(10, 6))
def plot_dist(dist, enhancement=False, figname='', **kwargs):
    if isinstance(dist, dict):
        Clean_line = dist['Clean']
        Transition_line = dist['Transition']
        Event_line = dist['Event']

        alpha = 0.8
        fig, ax = plt.subplots(1, 1)
        a, = ax.plot(dp, Clean_line, ls='solid', color='#1d4a9f', lw=3, alpha=alpha)
        b, = ax.plot(dp, Transition_line, ls='solid', color='#4a9f1d', lw=3, alpha=alpha)
        c, = ax.plot(dp, Event_line, ls='solid', color='#9f1d4a', lw=3, alpha=alpha)

        # Area
        ax.fill_between(dp, y1=0, y2=Clean_line, alpha=0.5, color='#84a7e9')
        ax.fill_between(dp, y1=Clean_line, y2=Transition_line, alpha=0.5, color='#a7e984')
        ax.fill_between(dp, y1=Transition_line, y2=Event_line, alpha=0.5, color='#e984a7')

        # figure_set
        xlim = kwargs.get('xlim') or (11.8, 2500)
        ylim = kwargs.get('ylim') or (0, 650)
        xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
        ylabel = kwargs.get('ylabel') or r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)

        if enhancement:
            ax2 = ax.twinx()
            d, = ax2.plot(dp, (Transition_line / Clean_line), ls='dashed', color='k', lw=3)
            e, = ax2.plot(dp, (Event_line / Transition_line), ls='dashed', color='gray', lw=3)
            # peak
            # arr1 = Transition_line / Clean_line
            # arr2 = Event_line / Transition_line
            # peaks1, _ = find_peaks(np.concatenate(([min(arr1)], arr1, [min(arr1)])), distance=20)
            # peaks2, _ = find_peaks(np.concatenate(([min(arr2)], arr2, [min(arr2)])), distance=20)
            # print(dp[peaks1-1])
            # print(dp[peaks2-1])

            plt.xlim(11.8, 2500)
            plt.ylim(0.5, 6)
            plt.xlabel(r'$\bf Diameter\ (nm)$')
            plt.ylabel(r'$\bf Enhancement\ ratio$')

            legend = ax.legend([a, b, c, d, e],
                               [r'$\bf Clean$', r'$\bf Transition$', r'$\bf Event$',
                                r'$\bf Number\ Enhancement\ 1$', r'$\bf Number\ Enhancement\ 2$'],
                               loc='upper left')
        else:
            legend = ax.legend([a, b, c],
                               [r'$\bf Clean$', r'$\bf Transition$', r'$\bf Event$'],
                               loc='upper left')

    title = kwargs.get('title') or ''
    plt.title(title, family='Times New Roman', weight='bold', size=20)
    plt.semilogx()
    plt.show()
    # fig.savefig(PATH_MAIN.parent / 'dist_plot' / f'{figname}')


@set_figure(figsize=(8, 6), fs=16)
def plot_dist_example(PSSD, PSSD_std, Q_ext, PESD, PESD_std, **kwargs):
    fig, ax = plt.subplots(1, 1)
    a, = ax.plot(dp, PSSD, ls='solid', color='#14336d', lw=2)
    ax.fill_between(dp, y1=PSSD-PSSD_std, y2=PSSD+PSSD_std, alpha=0.8, color='#d4ecf8')
    plt.grid(color='k', axis='x', which='major', linestyle='--', linewidth=0.25, alpha=0.2)
    # figure_set
    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, PSSD.max()*2)
    xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or r'$\bf dS/dlogdp\ (nm^{2}\ cm^{-3})$'
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    ax.spines['left'].set_color(a.get_color())
    ax.set_ylabel(r'$\bf dS/dlogdp\ (nm^{2}\ cm^{-3})$', c=a.get_color())
    ax.tick_params(axis='y', color=a.get_color(), labelcolor=a.get_color())

    ax2 = ax.twinx()
    b, = ax2.plot(dp, PESD, ls='solid', color='k', lw=2)
    ax2.fill_between(dp, y1=PESD-PESD_std, y2=PESD+PESD_std, alpha=0.8, color='#ede9e8')

    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, PESD.max() * 2)
    ylabel = r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'
    ax2.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    ax2.spines['right'].set_color(b.get_color())
    ax2.set_ylabel(r'$\bf d{\sigma}/dlogdp\ (1/Mm)$', c=b.get_color())
    ax2.tick_params(axis='y', color=b.get_color(), labelcolor=b.get_color())

    ax3 = ax.twinx()
    c, = ax3.plot(dp, Q_ext, ls='dashed', color='r', lw=2)
    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, Q_ext.max() * 1.5)
    ylabel = r'$\bf Q_{ext}$'
    ax3.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax3.spines['right'].set_position(('outward', 70))

    ax3.spines['right'].set_color(c.get_color())
    ax3.set_ylabel(r'$\bf Q_{ext}$', c=c.get_color())
    ax3.tick_params(axis='y', color=c.get_color(), labelcolor=c.get_color())


    legend = ax.legend([a, b, c],
                       [r'$\bf Surface\ dist.$', r'$\bf Extinction\ dist.$', r'$\bf Q_{ext}$'],
                       loc='upper left')
    title = kwargs.get('title') or ''
    plt.title(title, family='Times New Roman', weight='bold', size=20)
    plt.semilogx()
    plt.show()


@set_figure(figsize=(8, 6))
def plot_dry_amb_dist_test(dry_dp, ndp, dry_ndp, new_dry_ndp, **kwargs):
    fig, ax = plt.subplots(1, 1)
    widths = np.diff(dp)
    widths = np.append(widths, widths[-1])

    width2 = np.diff(dry_dp)
    width2 = np.append(width2, width2[-1])

    ax.bar(dp, ndp, width=widths, alpha=0.3)
    # ax.bar(dry_dp[:np.size(dry_ndp)], dry_ndp, width=width2[:np.size(dry_ndp)], color='r', alpha=0.3)
    ax.bar(dp, new_dry_ndp, width=widths, color='g', alpha=0.3)

    plt.semilogx()

    ax.plot(dp, ndp, ls='solid', color='b', lw=3)
    ax.plot(dp[:np.size(dry_ndp)], dry_ndp, ls='solid', color='r', lw=3)

    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, 2e5)
    xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or r'$\bf dN/dlogdp\ (1/Mm)$'
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    plt.show()



@set_figure(figsize=(8, 6))
def plot_dist_with_STD(Ext_amb_dis, Ext_amb_dis_std, Ext_dry_dis, Ext_dry_dis_std, state='Clean', **kwargs):
    PESD, PESD_std= Ext_amb_dis[state], Ext_amb_dis_std[state]
    PESD_std = np.array(pd.DataFrame(PESD_std).ewm(span=5).mean()).reshape(_length,)
    PESD_low, PESD_up = PESD - PESD_std, PESD + PESD_std


    PESD_dry, PESD_std_dry= Ext_dry_dis[state], Ext_dry_dis_std[state]
    PESD_std_dry = np.array(pd.DataFrame(PESD_std_dry).ewm(span=5).mean()).reshape(_length,)
    PESD_low_dry, PESD_up_dry = PESD_dry - PESD_std_dry, PESD_dry + PESD_std_dry

    fig, ax = plt.subplots(1, 1)
    a, = ax.plot(dp, PESD, ls='solid', color=color_choose[state][0], lw=2)
    b, = ax.plot(dp, PESD_dry, ls='dashed', color='k', lw=2)
    c = ax.fill_between(dp, y1=PESD_low, y2=PESD_up, alpha=0.5, color=color_choose[state][1], edgecolor=None)
    d = ax.fill_between(dp, y1=PESD_low_dry, y2=PESD_up_dry, alpha=0.5, color='#ece8e7', edgecolor=None)
    plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    # figure_set
    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, 850)
    xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    legend = ax.legend([a, b,],
                       [r'$\bf Ambient$', r'$\bf Dry$'],
                       loc='upper left')
    title = kwargs.get('title') or r'$\bf Extinction\ Distribution$'
    plt.title(title, family='Times New Roman', weight='bold', size=20)
    plt.semilogx()
    plt.show()
    # fig.savefig(PATH_MAIN.parent / 'dist_plot' / f'{state}_Ext_dist', transparent=True)


@set_figure(figsize=(8, 6))
def plot_dist2(dist, dist2, figname='', **kwargs):
    if isinstance(dist, dict):
        Clean_line = dist['Clean']
        Transition_line = dist['Transition']
        Event_line = dist['Event']

        Clean_line2 = dist2['Clean']
        Transition_line2 = dist2['Transition']
        Event_line2 = dist2['Event']

        alpha = 0.8
        fig, ax = plt.subplots(1, 1)
        a, = ax.plot(dp, Clean_line, ls='dotted', color='k', lw=3, alpha=alpha)
        b, = ax.plot(dp, Transition_line, ls='dashed', color='k', lw=3, alpha=alpha)
        c, = ax.plot(dp, Event_line, ls='solid', color='k', lw=3, alpha=alpha)

        # figure_set
        xlim = kwargs.get('xlim') or (11.8, 2500)
        ylim = kwargs.get('ylim') or (0, 2e5)
        xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
        ylabel = kwargs.get('ylabel') or r'$\bf dN/dlogdp $'
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2))
        plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)

        ax2 = ax.twinx()
        d, = ax2.plot(dp, Clean_line2, ls='dotted', color='b', lw=3, alpha=alpha)
        e, = ax2.plot(dp, Transition_line2, ls='dashed', color='b', lw=3, alpha=alpha)
        f, = ax2.plot(dp, Event_line2, ls='solid', color='b', lw=3, alpha=alpha)
        ylim = kwargs.get('ylim') or (0, 1.5e9)
        ylabel = kwargs.get('ylabel') or r'$\bf dS/dlogdp$'
        ax2.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        ax2.tick_params(axis='y', colors=d.get_color())
        ax2.tick_params(axis='y', colors=d.get_color())
        ax2.yaxis.label.set_color(d.get_color())
        ax2.yaxis.label.set_color(d.get_color())
        ax2.spines['right'].set_color(d.get_color())
        ax2.spines['right'].set_color(d.get_color())

        legend = ax.legend([a, b, c],
                           [r'$\bf Clean$', r'$\bf Transition$', r'$\bf Event$'],
                           loc='upper left')

    title = kwargs.get('title') or ''
    plt.title(title, family='Times New Roman', weight='bold', size=20)
    plt.semilogx()
    plt.show()
    # fig.savefig(PATH_MAIN.parent / 'dist_plot' / f'{figname}')


@set_figure(figsize=(8, 6))
def plot_dist_fRH(dist, dist2, figname='', **kwargs):
    if isinstance(dist, dict):
        cut = 10
        Clean_line = dist['Clean'][:-cut]
        Transition_line = dist['Transition'][:-cut]
        Event_line = dist['Event'][:-cut]

        Clean_line2 = dist2['Clean'][:-cut]
        Transition_line2 = dist2['Transition'][:-cut]
        Event_line2 = dist2['Event'][:-cut]

        alpha = 0.8
        fig, ax = plt.subplots(1, 1)

        a, = ax.plot(dp[:-cut], Clean_line2/Clean_line, ls='dotted', color='k', lw=3, alpha=alpha)
        b, = ax.plot(dp[:-cut], Transition_line2/Transition_line, ls='dashed', color='k', lw=3, alpha=alpha)
        c, = ax.plot(dp[:-cut], Event_line2/Event_line, ls='solid', color='k', lw=3, alpha=alpha)

        # figure_set
        xlim = kwargs.get('xlim') or (11.8, 2500)
        ylim = kwargs.get('ylim') or (0, 4)
        xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
        ylabel = kwargs.get('ylabel') or r'$\bf dN/dlogdp $'
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2))
        plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)


    title = kwargs.get('title') or ''
    plt.title(title, family='Times New Roman', weight='bold', size=20)
    plt.semilogx()
    plt.show()
    # fig.savefig(PATH_MAIN.parent / 'dist_plot' / f'{figname}')


@set_figure(figsize=(8, 6))
def plot_dist_cp(dist, std1, dist2, std2, figname='', **kwargs):
    PESD, PESD_std = dist, std1
    PESD_std = np.array(pd.DataFrame(PESD_std).ewm(span=5).mean()).reshape(_length, )*0.2
    PESD_low, PESD_up = PESD - PESD_std, PESD + PESD_std

    PESD_dry, PESD_std_dry = dist2, std2
    PESD_std_dry = np.array(pd.DataFrame(PESD_std_dry).ewm(span=5).mean()).reshape(_length, )*0.2
    PESD_low_dry, PESD_up_dry = PESD_dry - PESD_std_dry, PESD_dry + PESD_std_dry

    # 创建两个数组
    approximate = np.array(dist)
    exact = np.array(dist2)
    difference = np.subtract(approximate, exact)
    abs_difference = np.absolute(difference)
    percentage_error = np.divide(abs_difference, exact) * 100
    percentage_error = np.array(pd.DataFrame(percentage_error).ewm(span=10).mean()).reshape(_length, )
    print(percentage_error)

    fig, ax = plt.subplots(1, 1)
    a, = ax.plot(dp, PESD, ls='solid', color=color_choose['Clean'][0], lw=2)
    b, = ax.plot(dp, PESD_dry, ls='solid', color=color_choose['Transition'][0], lw=2)
    c = ax.fill_between(dp, y1=PESD_low, y2=PESD_up, alpha=0.3, color=color_choose['Clean'][0], edgecolor=None)
    d = ax.fill_between(dp, y1=PESD_low_dry, y2=PESD_up_dry, alpha=0.3, color=color_choose['Transition'][0], edgecolor=None)
    plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)
    # figure_set
    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, 600)
    xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    plt.semilogx()

    ax2 = ax.twinx()
    c = ax2.scatter(dp, percentage_error, color='white', edgecolor='k')
    ax2.set_ylabel(r'$\bf Error\ (\%)$')

    legend = ax.legend([a, b, c],
                       [r'$\bf Internal$', r'$\bf External$', r'$\bf Error$'],
                       loc='upper left')
    title = kwargs.get('title') or r'$\bf Extinction\ Distribution$'
    plt.title(title, family='Times New Roman', weight='bold', size=20)

    plt.show()
    # fig.savefig(PATH_MAIN.parent / 'dist_plot' / f'000_Ext_dist', transparent=True)
    return fig, ax
