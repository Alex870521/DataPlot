import numpy as np
import pandas as pd
from DataPlot import *

PNSD = DataReader('PNSD_dNdlogdp.csv')
PSSD = DataReader('PSSD_dSdlogdp.csv')
PVSD = DataReader('PVSD_dVdlogdp.csv')
PESD_inter = DataReader('PESD_dextdlogdp_internal.csv')
PESD_dry_inter = DataReader('PESD_dextdlogdp_dry_internal.csv')
PESD_exter = DataReader('PESD_dextdlogdp_external.csv')
df = DataBase

Ext_amb_dis_internal, Ext_amb_dis_std_internal = DataClassifier(PESD_inter, by='State')
Ext_dry_dis_internal, Ext_dry_dis_std_internal = DataClassifier(PESD_dry_inter, by='State')
Ext_amb_dis_external, Ext_amb_dis_std_external = DataClassifier(PESD_exter, by='State')

PNSD_amb_dis, PNSD_amb_dis_std = DataClassifier(PNSD, by='State')
PSSD_amb_dis, PSSD_amb_dis_std = DataClassifier(PSSD, by='State')
PVSD_amb_dis, PVSD_amb_dis_std = DataClassifier(PVSD, by='State')

grp_, _ = DataClassifier(PESD_inter, by='Extinction', qcut=True, q=10, statistic='Table')


if __name__ == '__main__':
    dp = SizeDist().dp
    # plot.overlay_dist(dp, Ext_amb_dis_internal, enhancement=True)
    # plot.separate_dist(dp, PNSD_amb_dis, PSSD_amb_dis, PVSD_amb_dis)
    # plot.heatmap(PNSD)
    # plot.heatmap(PSSD)
    # plot.heatmap(PVSD)
    # plot.heatmap(PESD_inter)
    # plot.dist_with_std(dp, Ext_amb_dis_internal, Ext_amb_dis_std_internal, Ext_dry_dis_internal, Ext_dry_dis_std_internal)
    # plot.compare(dp, Ext_amb_dis_internal['Event'], Ext_amb_dis_std_internal['Event'], Ext_amb_dis_external['Event'], Ext_amb_dis_std_external['Event'])
    # plot.curve_fitting(dp, Ext_amb_dis_internal['Transition'], mode=3)
    plot.three_dimension(dp, grp_, weighting='PESD')
