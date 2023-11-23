import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas import read_csv, concat
from PyMieScatt.Mie import AutoMieQ
from DataPlot.Data_processing import main
from DataPlot.Data_processing.Data_classify import state_classify
from DataPlot.plot_templates import set_figure, unit
from pathlib import Path
from DataPlot.plot_templates import set_figure, unit, getColor
from scipy.signal import find_peaks


PATH_MAIN = Path(__file__).parent.parent / 'Data'
PATH_DIST = PATH_MAIN / 'Level2' / 'distribution'


with open(PATH_DIST / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PSSD_dSdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PSSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PVSD_dVdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PVSD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PESD_dEdlogdp_internal.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PESD = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PESDist_dry.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PESD_dry = read_csv(f, parse_dates=['Time']).set_index('Time')

with open(PATH_DIST / 'PESD_dEdlogdp_external.csv', 'r', encoding='utf-8', errors='ignore') as f:
    PESD_external = read_csv(f, parse_dates=['Time']).set_index('Time')

dp = np.array(PNSD.columns, dtype='float')
_length = np.size(dp)
dlogdp = np.array([0.014] * _length)

df = main()
state_classify(df)

Ext_amb_df = concat([df[['Extinction', 'State']], PESD], axis=1)
Ext_dry_df = concat([df[['Extinction', 'State']], PESD_dry], axis=1)
Ext_amb_df_external = concat([df[['Extinction', 'State']], PESD_external], axis=1)

PSD_amb_df = concat([df[['Extinction', 'State']], PNSD], axis=1)
PSSD_amb_df = concat([df[['Extinction', 'State']], PSSD], axis=1)
PVSD_amb_df = concat([df[['Extinction', 'State']], PVSD], axis=1)

if __name__ == '__main__':
    print('')
    # plot_dist(Ext_amb_dis, title=r'$\bf Ambient\ Extinction\ Distribution$', enhancement=False, figname='Amb_Ext_Dist')
    # plot_dist(Ext_dry_dis, title=r'$\bf Dry\ Extinction\ Distribution$', enhancement=True, figname='Dry_Ext_Dist')
    # plot_dist(PSD_amb_dis, ylim=(0, 1.5e5), ylabel=r'$\bf dN/dlogdp\ (1/Mm)$', title=r'$\bf Ambient\ Particle\ Number\ Distribution$')

    # plot_dist2(PSD_amb_dis, PSSD_amb_dis, title=r'$\bf Particle\ Number\ &\ Surface\ Area\ Distribution$', figname='NumSurf_dist')
    # plot_dist_fRH(Ext_dry_dis, Ext_amb_dis, title=r'$\bf Distribution$', figname='fRH_dist')
    # plot_dist_example(PSSD_amb_df.mean()[1:], PSSD_amb_df.std()[1:], Q_ext, Ext_amb_df.mean()[1:], Ext_amb_df.std()[1:])

    # plot_dist_with_STD(Ext_amb_dis, Ext_amb_dis_std, Ext_dry_dis, Ext_dry_dis_std, state='Clean')
    # plot_dist_with_STD(Ext_amb_dis, Ext_amb_dis_std, Ext_dry_dis, Ext_dry_dis_std, state='Transition')
    # plot_dist_with_STD(Ext_amb_dis, Ext_amb_dis_std, Ext_dry_dis, Ext_dry_dis_std, state='Event')

    # dist1, diat_std1 = Ext_amb_df.dropna().iloc[:, 2:].mean(),  Ext_amb_df.dropna().iloc[:, 2:].std()
    # dist2, diat_std2 = Ext_amb_df_external.dropna().iloc[:, 2:].mean(),  Ext_amb_df_external.dropna().iloc[:, 2:].std()
    # fig, ax = plot_dist_cp(dist1, diat_std1, dist2, diat_std2)