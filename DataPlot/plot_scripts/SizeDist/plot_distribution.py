import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataPlot.plot_templates import set_figure
from DataPlot.Data_processing import SizeDist


color_choose = {'Clean': ['#1d4a9f', '#84a7e9'],
                'Transition': ['#4a9f1d', '#a7e984'],
                'Event': ['#9f1d4a', '#e984a7']}

dp = SizeDist().dp


@set_figure(figsize=(6, 6))
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
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useMathText=True)

        if enhancement:
            ax2 = ax.twinx()
            d, = ax2.plot(dp, (Transition_line / Clean_line), ls='dashed', color='k', lw=3)
            e, = ax2.plot(dp, (Event_line / Transition_line), ls='dashed', color='gray', lw=3)

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



@set_figure(figsize=(6, 6))
def plot_dist_with_STD(Ext_amb_dis, Ext_amb_dis_std, Ext_dry_dis, Ext_dry_dis_std, state='Clean', **kwargs):
    fig, ax = plt.subplots(1, 1)
    for state in Ext_amb_dis.keys():

        PESD, PESD_std= Ext_amb_dis[state], Ext_amb_dis_std[state]
        PESD_std = np.array(pd.DataFrame(PESD_std).ewm(span=5).mean()).reshape(167,)
        PESD_low, PESD_up = PESD - PESD_std, PESD + PESD_std

        PESD_dry, PESD_std_dry= Ext_dry_dis[state], Ext_dry_dis_std[state]
        PESD_std_dry = np.array(pd.DataFrame(PESD_std_dry).ewm(span=5).mean()).reshape(167,)
        PESD_low_dry, PESD_up_dry = PESD_dry - PESD_std_dry, PESD_dry + PESD_std_dry

        a, = ax.plot(dp, PESD, ls='solid', color=color_choose[state][0], lw=2, label=f'Amb {state}')
        b, = ax.plot(dp, PESD_dry, ls='dashed', color=color_choose[state][1], lw=2, label=f'Dry {state}')
        c = ax.fill_between(dp, y1=PESD_low, y2=PESD_up, alpha=0.4, color=color_choose[state][1], edgecolor=None, label='__nolegend__')
        # d = ax.fill_between(dp, y1=PESD_low_dry, y2=PESD_up_dry, alpha=0.5, color='#ece8e7', edgecolor=color_choose[state][1], label='__nolegend__')
        plt.grid(color='k', axis='x', which='major', linestyle='dashdot', linewidth=0.4, alpha=0.4)


    # figure_set
    xlim = kwargs.get('xlim') or (11.8, 2500)
    ylim = kwargs.get('ylim') or (0, 850)
    xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
    ylabel = kwargs.get('ylabel') or r'$\bf d{\sigma}/dlogdp\ (1/Mm)$'
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    legend = ax.legend(loc='upper left', prop={'weight': 'bold'})
    title = kwargs.get('title') or r'$\bf Extinction\ Distribution$'
    plt.title(title, family='Times New Roman', weight='bold', size=20)
    plt.semilogx()
    plt.show()
    # fig.savefig(PATH_MAIN.parent / 'dist_plot' / f'{state}_Ext_dist', transparent=True)





# dealing
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
def compare(dist, std1, dist2, std2, ax=None, **kwargs):
    PESD, PESD_std = dist, std1
    PESD_std = np.array(pd.DataFrame(PESD_std).ewm(span=5).mean()).reshape(167, )*0.2
    PESD_low, PESD_up = PESD - PESD_std, PESD + PESD_std

    PESD_dry, PESD_std_dry = dist2, std2
    PESD_std_dry = np.array(pd.DataFrame(PESD_std_dry).ewm(span=5).mean()).reshape(167, )*0.2
    PESD_low_dry, PESD_up_dry = PESD_dry - PESD_std_dry, PESD_dry + PESD_std_dry

    # 创建两个数组
    approximate = np.array(dist)
    exact = np.array(dist2)
    difference = np.subtract(approximate, exact)
    abs_difference = np.absolute(difference)
    percentage_error = np.divide(abs_difference, exact) * 100
    percentage_error = np.array(pd.DataFrame(percentage_error).ewm(span=10).mean()).reshape(167, )


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

    return ax
