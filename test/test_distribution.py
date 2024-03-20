import numpy as np
from DataPlot import *

PNSD = DataReader('PNSD_dNdlogdp.csv')
PSSD = DataReader('PSSD_dSdlogdp.csv')
PVSD = DataReader('PVSD_dVdlogdp.csv')
PESD_inter = DataReader('PESD_dextdlogdp_internal.csv')
PESD_dry_inter = DataReader('PESD_dextdlogdp_dry_internal.csv')
PESD_exter = DataReader('PESD_dextdlogdp_external.csv')

Ext_amb_dis_internal, Ext_amb_dis_std_internal = DataClassifier(PESD_inter, by='State', statistic='Table')
Ext_dry_dis_internal, Ext_dry_dis_std_internal = DataClassifier(PESD_dry_inter, by='State', statistic='Table')
Ext_amb_dis_external, Ext_amb_dis_std_external = DataClassifier(PESD_exter, by='State', statistic='Table')

PNSD_amb_dis, PNSD_amb_dis_std = DataClassifier(PNSD, by='State', statistic='Table')
PSSD_amb_dis, PSSD_amb_dis_std = DataClassifier(PSSD, by='State', statistic='Table')
PVSD_amb_dis, PVSD_amb_dis_std = DataClassifier(PVSD, by='State', statistic='Table')

ext_grp, _ = DataClassifier(PESD_inter, by='Extinction', statistic='Table', qcut=10)


if __name__ == '__main__':
    plot.heatmap(PNSD)
    # plot.heatmap(PSSD)
    # plot.heatmap(PVSD)
    # plot.heatmap(PESD_inter)
    # plot.overlay_dist(Ext_amb_dis_internal, diff="Error")
    # plot.separate_dist(PNSD_amb_dis, PSSD_amb_dis, PVSD_amb_dis)
    # plot.dist_with_std(Ext_amb_dis_internal, Ext_amb_dis_std_internal, std_scale=0.5)
    # plot.curve_fitting(np.array(Ext_amb_dis_internal.columns, dtype=float), Ext_amb_dis_internal.loc['Transition'], mode=4)
    plot.three_dimension(ext_grp, unit='Extinction')
