import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DataPlot.templates import set_figure, unit, getColor, color_maker
from pathlib import Path
from pandas import read_csv, concat, date_range
from DataPlot.data_processing import main
from DataPlot.data_processing.Data_classify import state_classify, season_classify, Seasons

PATH_MAIN = Path(__file__).parents[3] / 'Data-Code-example' / 'Level2' / 'distribution'

with open(PATH_MAIN / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_MAIN / 'PSSD_dSdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PSSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_MAIN / 'PVSD_dVdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PVSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_MAIN / 'PESD_dextdlogdp_internal.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PESD = read_csv(f, parse_dates=['Time']).set_index('Time')

# Time Series
df = main()
dic_grp_sta = state_classify(df)

# for i in ['Extinction', 'Scattering', 'Absorption', 'MEE', 'MSE', 'MAE']:
#     print(i)
# print(dic_grp_sta['Clean'][i].quantile([0.001, 0.999]))
# print('Mean=', dic_grp_sta['Clean'][i].mean().round(2), 'Std=', dic_grp_sta['Clean'][i].std().round(2))


for season, (st_tm_, fn_tm_) in Seasons.items():

    st_tm, fn_tm = pd.Timestamp(st_tm_), pd.Timestamp(fn_tm_)
    IdxTmRange = pd.date_range(st_tm, fn_tm, freq='1h')

    df_ = df.loc[st_tm:fn_tm].reindex(IdxTmRange)
    time_ = df_.index

    PNSD_data = PNSD.loc[st_tm:fn_tm].reindex(IdxTmRange)
    PSSD_data = PSSD.loc[st_tm:fn_tm].reindex(IdxTmRange)
    time = PNSD_data.index
    dp = PNSD.keys().astype(float)

    # 數據平滑
    df_ = df_.rolling(3).mean(numeric_only=True)
    # df_.Extinction = df_.Extinction.fillna(0)  # 使用0填充NaN值
    # df_.Extinction = df_.Extinction.replace([np.inf, -np.inf], 0)


    @set_figure(fs=12)
    def timeSeries2():
        fig, (ax1, ax2, ax3, ax6) = plt.subplots(4, 1, figsize=(12, 6))
        fig.subplots_adjust(right=1 - 0.1)
        sc_1, = ax1.plot(time_, df_.Extinction, lw=1.5, color="b", alpha=1)
        sc_2, = ax1.plot(time_, df_.Scattering, lw=1.5, color="g", alpha=1)
        sc_3, = ax1.plot(time_, df_.Absorption, lw=1.5, color="r", alpha=1)
        ax1.set_ylabel(r'$\bf b_{{ext, scat, abs}}\ (1/Mm)$')
        ax1.set(ylim=(0., df_.Extinction.max()*1.1), xlim=(st_tm, fn_tm))
        ax1.axes.xaxis.set_visible(False)
        ax1.legend([sc_1, sc_2, sc_3], [r'$\bf Extinction$', r'$\bf Scattering$', r'$\bf Absorption$'],
                   loc='upper right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, labelspacing=0.5, handlelength=1)

        # Temp, RH
        AT, = ax2.plot(time_, df_.AT, color="r", alpha=1)
        ax2.set_ylabel(r'$\bf Temp\ (^{\circ}C)$')
        ax2.set(ylim=(df_.AT.min() - 2, df_.AT.max() + 2), xlim=(st_tm, fn_tm))
        ax2.axes.xaxis.set_visible(False)
        ax2.tick_params(axis='y', colors=AT.get_color())
        ax2.yaxis.label.set_color(AT.get_color())
        ax2.spines['left'].set_color(AT.get_color())

        ax2_2 = ax2.twinx()
        RH, = ax2_2.plot(time_, df_.RH, color="b", alpha=1)
        ax2_2.set_ylabel(r'$\bf RH\ (\%)$')
        ax2_2.set(ylim=(20, 100), xlim=(st_tm, fn_tm))
        ax2_2.axes.xaxis.set_visible(False)
        ax2_2.tick_params(axis='y', colors=RH.get_color())
        ax2_2.yaxis.label.set_color(RH.get_color())
        ax2_2.spines['right'].set_color(RH.get_color())
        ax2_2.spines['left'].set_color(AT.get_color())



        scalar_map, colors = color_maker(df_.PBLH.values)
        ax3.bar(time_, df_.VC, color=scalar_map.to_rgba(colors), width=0.0417, edgecolor='None', linewidth=0)
        # ax3.bar(time_, np.where(colors == 0, df_.VC, 0), width=0, color='None', alpha=1, edgecolor='None', linewidth=0, zorder=2)
        ax3.set_ylabel(r'$\bf VC\ (m^2/s)$')
        ax3.set(ylim=(0, df_.VC.max()*1.1), xlim=(st_tm, fn_tm))
        ax3.axes.xaxis.set_visible(False)

        ax3_1 = inset_axes(ax3,
                           width="1%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax3.transAxes,
                           borderpad=0,
                           )

        cbar = plt.colorbar(scalar_map, cax=ax3_1, orientation='vertical', ticks=[0, 200, 400, 600, 800])
        cbar.set_label(label=r'$\bf PBLH (m)$')

        ax3_2 = ax3.twinx()
        sc_6 = ax3_2.scatter(time_, df_.WS, c=df_.WD, cmap='hsv', marker='o', s=5, alpha=1.0)
        ax3_2.set_ylabel(r'$\bf WS\ (m/s)$')
        ax3_2.set_ylim((-2, df_.WS.max()*1.1))
        ax3_2.set_yticks([0, 2, 4])

        ax3_3 = inset_axes(ax3, width="30%", height="5%", loc='upper left')
        color_bar2 = plt.colorbar(sc_6, cax=ax3_3, orientation='horizontal')
        color_bar2.set_label(label=r'$\bf WD $')


        sc_6 = ax6.scatter(time_, df_.PM25, c=df_.PM1 / df_.PM25, vmin=0.2, vmax=1, cmap='jet', marker='o', s=5, alpha=1.0)
        ax6.set_ylabel(r'$\bf PM_{2.5}\ (\mu g/m^3)$')
        ax6.set(ylim=(0, df_.PM25.max() * 1.2), xlim=(st_tm, fn_tm))
        ax4_2 = inset_axes(ax6, width="30%", height="5%", loc='upper left')
        color_bar2 = plt.colorbar(sc_6, cax=ax4_2, orientation='horizontal')
        color_bar2.set_label(label=r'$\bf PM_{1}/PM_{2.5} $')
        # fig.savefig(f'time2_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
        plt.show()


    # timeSeries()
    # timeSeries2()


    def extinction_timeseries(df):
        # 消光 PM1 PM2.5 整年度時序
        fig2, (axes1, axes2) = plt.subplots(2, 1, figsize=(12, 5), dpi=150, constrained_layout=True)
        sc = axes1.scatter(df.index, df.Extinction, c=df.PM25, norm=plt.Normalize(vmin=0, vmax=50), cmap='jet',
                           marker='o', s=10, facecolor="b", edgecolor=None, alpha=1)

        axes1.set_title('Extinction & $\mathregular{PM_{2.5}}$ Sequence Diagram')
        axes1.set_xlabel('')
        axes1.set_ylabel('Ext (1/Mm)')
        axes1.set_ylim(0., 600)
        axes1.set_xlim(18506., 18871.)

        axins = inset_axes(axes1, width="40%", height="5%", loc=1)
        color_bar = plt.colorbar(sc, cax=axins, orientation='horizontal')
        color_bar.set_label(label='$\mathregular{PM_{2.5}}$' + ' ($\mathregular{\mu}$g/$\mathregular{m^3}$)',
                            family='Times New Roman', weight='bold', size=14)

        color_bar.ax.set_xticklabels(color_bar.ax.get_xticks().astype(int), size=14)
        ###
        sc2 = axes2.scatter(df.index, df.Extinction, c=df.PM1, norm=plt.Normalize(vmin=0, vmax=30), cmap='jet',
                            marker='o', s=10, facecolor="b", edgecolor=None, alpha=1)

        axes2.set_title('Extinction & $\mathregular{PM_{1.0}}$ Sequence Diagram')
        axes2.set_xlabel('')
        axes2.set_ylabel('Ext (1/Mm)')
        axes2.set_ylim(0., 600)
        axes2.set_xlim(18506., 18871.)

        axins2 = inset_axes(axes2, width="40%", height="5%", loc=1)
        color_bar2 = plt.colorbar(sc2, cax=axins2, orientation='horizontal')
        color_bar2.set_label(label='$\mathregular{PM_{1.0}}$' + ' ($\mathregular{\mu}$g/$\mathregular{m^3}$)',
                             family='Times New Roman', weight='bold', size=14)
        color_bar2.ax.set_xticklabels(color_bar2.ax.get_xticks().astype(int), size=14)
        plt.show()

    def extinction_month(df_group_season):
        # 消光散光吸光逐月or逐季 時序
        for _key, _df in df_group_season:
            print(f'Plot : {_df.Season[0]}')
            st_tm, fn_tm = _df.index[0], _df.index[-1]
            tick_time = date_range(st_tm, fn_tm, freq='10d')  ## set tick

            fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(11, 5), dpi=150)

            sc1 = ax1.scatter(_df.index, _df.Scatter,
                              marker='o', s=15, facecolor="g", edgecolor='k', linewidths=0.3, alpha=0.9)
            sc2 = ax1.scatter(_df.index, _df.Absorption,
                              marker='o', s=15, facecolor="r", edgecolor='k', linewidths=0.3, alpha=0.9)

            ax1.set_xlabel('Date')
            ax1.set_ylabel('Scattering & \n Absorption (1/Mm)')
            ax1.set_xticks(tick_time)
            ax1.set_xticklabels(tick_time.strftime('%m/%d'))
            ax1.set_ylim(0, _df.Scatter.max() + 10)
            ax1.set_xlim(st_tm, fn_tm)
            [ax1.spines[axis].set_visible(False) for axis in ['top']]

            sc3 = ax2.scatter(_df.index, _df.Extinction,
                              marker='o', s=15, facecolor="b", edgecolor='k', linewidths=0.3, alpha=0.9)
            therosold = ax2.plot(_df.index, np.full(len(_df.index), _df.Extinction.quantile([0.90])), color='r',
                                 ls='--', lw=2)
            ax2.set_ylabel('Extinction (1/Mm)', loc='bottom')
            ax2.set_xticks(tick_time)
            ax2.set_xticklabels(tick_time.strftime(''))
            ax2.set_ylim(0, _df.Scatter.max() + 10)
            ax2.set_xlim(st_tm, fn_tm)

            ax2.set_title(str(_df.Season[0]) + ' Sequence Diagram')

            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()

            [ax2.spines[axis].set_visible(False) for axis in ['bottom']]
            ax2.get_xaxis().set_visible(False)

            ax2.legend(handles=[sc1, sc2, sc3], labels=['Scattering', 'Absorption', 'Extinction'],
                       bbox_to_anchor=(0, 1.), loc='upper left')
            plt.subplots_adjust(hspace=0.0)
            # fig.savefig(pth(f"Optical{_df.Season[0]}"))
            plt.show()

if __name__ == '__main__':
    print('0')
    # timeSeries2()
    # extinction_timeseries(df)

