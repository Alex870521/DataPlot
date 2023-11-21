from pathlib import Path
from pandas import read_csv, concat
from Data_processing import integrate
from Data_classify import state_classify, season_classify, Seasons
from datetime import datetime
import pandas as pd
import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_templates import set_figure, unit, getColor, color_maker

PATH_MAIN = Path("C:/Users/alex/PycharmProjects/DataPlot/Data")
PATH_DIST = Path("C:/Users/alex/PycharmProjects/DataPlot/Data/Level2/distribution")

with open(PATH_DIST / 'PNSDist.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PSSDist.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PSSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PVSDist.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PVSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PESDist.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PESD = read_csv(f, parse_dates=['Time']).set_index('Time')

# Time Series
df = integrate()
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
    df_ = df_.rolling(3).mean()
    # df_.Extinction = df_.Extinction.fillna(0)  # 使用0填充NaN值
    # df_.Extinction = df_.Extinction.replace([np.inf, -np.inf], 0)

    @setFigure(fs=12)
    def timeSeries():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 3))
        pco1 = ax1.pcolormesh(time, dp, PNSD_data.interpolate(limit=2).T,
                              cmap='jet',
                              shading='auto',
                              norm=colors.PowerNorm(gamma=0.6, vmax=PNSD_data.max(axis=0).quantile(0.8)))
        ax1.set(yscale='log', ylim=(11.8, 1000))
        ax1.set_ylabel(r'$\bf dp\ (nm)$', fontsize=12, weight='bold', )
        ax1.axes.xaxis.set_visible(False)

        cbar = plt.colorbar(pco1, ax=ax1, pad=0.01)
        cbar.set_label(r'$\bf dN/dlogdp$', fontsize=12, weight='bold', labelpad=5)
        cbar.ax.ticklabel_format(axis='y', scilimits=(-2, 3), useMathText=True)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.ax.yaxis.offsetText.set_fontproperties(dict(size=12))

        pco2 = ax2.pcolormesh(time, dp, PSSD_data.interpolate(limit=2).T,
                              cmap='jet',
                              shading='auto',
                              norm=colors.PowerNorm(gamma=0.6, vmax=PSSD_data.max(axis=0).quantile(0.8)))
        # ax2.config(time, PSSD_data.idxmax(axis=0).ewm(span=6).mean(), ls='dashed', lw=1.5, color='gray')
        ax2.set(yscale='log', ylim=(11.8, 2500))
        ax2.set_ylabel(r'$\bf dp\ (nm)$')
        ax2.axes.xaxis.set_visible(False)

        cbar2 = plt.colorbar(pco2, ax=ax2, pad=0.01)
        cbar2.set_label(r'$\bf dS/dlogdp$', fontsize=12, weight='bold', )
        cbar2.ax.ticklabel_format(axis='y', scilimits=(-2, 3), useMathText=True)
        cbar2.ax.yaxis.set_offset_position('left')
        cbar2.ax.yaxis.offsetText.set_fontproperties(dict(size=12))
        fig.savefig(f'time1_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
        plt.show()


    @setFigure(fs=12)
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
        fig.savefig(f'time2_{st_tm.strftime("%Y%m%d")}_{fn_tm.strftime("%Y%m%d")}.png')
        plt.show()


    timeSeries()
    timeSeries2()

if __name__ == '__main__':
    print('0')
    # timeSeries()
    # timeSeries2()
