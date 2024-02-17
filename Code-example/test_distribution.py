import numpy as np
from pandas import concat
from DataPlot import *


PNSD = DataReader('PNSD_dNdlogdp.csv')
PSSD = DataReader('PSSD_dSdlogdp.csv')
PVSD = DataReader('PVSD_dVdlogdp.csv')
PESD_inter = DataReader('PESD_dextdlogdp_internal.csv')
PESD_dry_inter = DataReader('PESD_dextdlogdp_dry_internal.csv')
PESD_exter = DataReader('PESD_dextdlogdp_external.csv')
df = DataReader('All_data.csv')


Classifier(df, 'state')

Ext_amb_df_internal = concat([df[['Extinction', 'State']], PESD_inter], axis=1)
Ext_dry_df = concat([df[['Extinction', 'State']], PESD_dry_inter], axis=1)
Ext_amb_df_external = concat([df[['Extinction', 'State']], PESD_exter], axis=1)
#
PNSD_amb_df = concat([df[['Extinction', 'State']], PNSD], axis=1)
PSSD_amb_df = concat([df[['Extinction', 'State']], PSSD], axis=1)
PVSD_amb_df = concat([df[['Extinction', 'State']], PVSD], axis=1)


def get_statistic(group):
    _avg, _std = {}, {}
    for name, subdf in group:
        _avg[name] = np.array(subdf.mean(numeric_only=True)[1:])
        _std[name] = np.array(subdf.std(numeric_only=True)[1:])
    return _avg, _std


Ext_amb_dis_internal, Ext_amb_dis_std_internal = get_statistic(Ext_amb_df_internal.dropna().groupby('State'))
Ext_dry_dis_internal, Ext_dry_dis_std_internal = get_statistic(Ext_dry_df.dropna().groupby('State'))
Ext_amb_dis_external, Ext_amb_dis_std_external = get_statistic(Ext_amb_df_external.dropna().groupby('State'))

PNSD_amb_dis, PNSD_amb_dis_std = get_statistic(PNSD_amb_df.dropna().groupby('State'))
PSSD_amb_dis, PSSD_amb_dis_std = get_statistic(PSSD_amb_df.dropna().groupby('State'))
PVSD_amb_dis, PVSD_amb_dis_std = get_statistic(PVSD_amb_df.dropna().groupby('State'))


if __name__ == '__main__':
    dp = SizeDist().dp
    # plot.overlay_dist(dp, Ext_amb_dis_internal, enhancement=True)
    # plot.separate_dist(dp, PNSD_amb_dis, PSSD_amb_dis, PVSD_amb_dis)
    # plot.heatmap(PNSD)
    # plot.heatmap(PSSD)
    # plot.heatmap(PVSD)
    plot.heatmap(PESD_inter)
    # plot.dist_with_std(dp, Ext_amb_dis_internal, Ext_amb_dis_std_internal, Ext_dry_dis_internal, Ext_dry_dis_std_internal)
    #
    # dist1, diat_std1 = Ext_amb_df_internal.dropna().iloc[:, 2:].mean(),  Ext_amb_df_internal.dropna().iloc[:, 2:].std()
    # dist2, diat_std2 = Ext_amb_df_external.dropna().iloc[:, 2:].mean(),  Ext_amb_df_external.dropna().iloc[:, 2:].std()
    # plot.compare(dp, dist1, diat_std1, dist2, diat_std2)
    #
    # plot.curvefit(dp, Ext_amb_dis_internal['Transition'], mode=3)