import numpy as np
from pathlib import Path
from pandas import concat
from DataPlot.Data_processing import main
from DataPlot.Data_processing.Data_classify import state_classify
from DataPlot.Data_processing import sizedist_reader, extdist_reader, dry_extdist_reader
from DataPlot.plot_scripts.SizeDist.plot import *
from DataPlot.plot_scripts.SizeDist.curve_fitting import *
from DataPlot.plot_scripts.SizeDist.plot_distribution import *

PATH_MAIN = Path(__file__).parent.parent / 'Data'
PATH_DIST = PATH_MAIN / 'Level2' / 'distribution'

PNSD, PSSD, PVSD = sizedist_reader()

PESD_inter, PESD_exter = extdist_reader()

PESD_dry_inter = dry_extdist_reader()

df = main()
state_classify(df)

Ext_amb_df_internal = concat([df[['Extinction', 'State']], PESD_inter], axis=1)
Ext_dry_df = concat([df[['Extinction', 'State']], PESD_dry_inter], axis=1)
Ext_amb_df_external = concat([df[['Extinction', 'State']], PESD_exter], axis=1)

PNSD_amb_df = concat([df[['Extinction', 'State']], PNSD], axis=1)
PSSD_amb_df = concat([df[['Extinction', 'State']], PSSD], axis=1)
PVSD_amb_df = concat([df[['Extinction', 'State']], PVSD], axis=1)


def get_statistic(group):
    _avg, _std = {}, {}
    for name, subdf in group:
        _avg[name] = np.array(subdf.mean(numeric_only=True)[1:])
        _std[name] = np.array(subdf.std(numeric_only=True)[1:])
    return _avg, _std


Ext_amb_dis, Ext_amb_dis_std = get_statistic(Ext_amb_df_internal.dropna().groupby('State'))
Ext_dry_dis, Ext_dry_dis_std = get_statistic(Ext_dry_df.dropna().groupby('State'))
Ext_amb_dis_external, Ext_amb_dis_std_external = get_statistic(Ext_amb_df_external.dropna().groupby('State'))

PNSD_amb_dis, PNSD_amb_dis_std = get_statistic(PNSD_amb_df.dropna().groupby('State'))
PSSD_amb_dis, PSSD_amb_dis_std = get_statistic(PSSD_amb_df.dropna().groupby('State'))
PVSD_amb_dis, PVSD_amb_dis_std = get_statistic(PVSD_amb_df.dropna().groupby('State'))


if __name__ == '__main__':
    plot_dist(Ext_amb_dis, title=r'$\bf Ambient\ Extinction\ Distribution$', enhancement=False, figname='Amb_Ext_Dist')
    plot_dist(PSSD_amb_dis, ylim=(0, 1.5e5), ylabel=r'$\bf dN/dlogdp\ (1/Mm)$', title=r'$\bf Ambient\ Particle\ Number\ Distribution$')

    plot_dist_with_STD(Ext_amb_dis, Ext_amb_dis_std, Ext_dry_dis, Ext_dry_dis_std)

    dist1, diat_std1 = Ext_amb_df_internal.dropna().iloc[:, 2:].mean(),  Ext_amb_df_internal.dropna().iloc[:, 2:].std()
    dist2, diat_std2 = Ext_amb_df_external.dropna().iloc[:, 2:].mean(),  Ext_amb_df_external.dropna().iloc[:, 2:].std()
    fig, ax = compare(dist1, diat_std1, dist2, diat_std2)



    dp = np.array(PNSD.columns, dtype='float')

    # curvefit(dp, Ext_amb_dis['Transition'], mode=10, figname='ext_trans')
    # heatmap(PNSD.index, dp, PNSD)

    # plot_NSV_dist(PNSD_amb_dis, PSSD_amb_dis, PVSD_amb_dis, title=r'$\bf Particle\ Size\ Distribution$', figname='NumSurf_dist')
