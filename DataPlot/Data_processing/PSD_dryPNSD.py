import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_csv, concat
from DataPlot.Data_processing.csv_decorator import save_to_csv
from DataPlot.Data_processing._reader import psd_reader, chemical_reader


PATH_MAIN = Path(__file__).parent.parent.parent / 'Data' / 'Level2'
PATH_DIST = PATH_MAIN / 'distribution'


PNSD = psd_reader()
chemical = chemical_reader()

df = concat([PNSD(), chemical()], axis=1)


dp = np.array(PNSD.columns, dtype='float')
_length = np.size(dp)
dlogdp = np.array([0.014] * _length)


@save_to_csv(PATH_DIST / 'PNSDist_dry.csv')
def dry_PNSD_process(**kwargs):
    index = df.index.copy()
    df_input = df.dropna()
    _index = df_input.index.copy()

    out_dis = {'dry_dist': [],
               }

    for _tm, _ser in df_input.iterrows():
        ndp = np.array(_ser[:_length])
        # uniform gRH
        gRH = resloved_gRH(_length, _ser['gRH'], uniform=True)

        dry_dp = dp / gRH
        belong_which_ibin = np.digitize(dry_dp, dp) - 1

        result = {}
        for i, (ibin, dn) in enumerate(zip(belong_which_ibin, ndp)):
            if dp[ibin] not in result:
                result[dp[ibin]] = []
            result[dp[ibin]].append(ndp[i])

        dry_ndp = []
        for key, val in result.items():
            dry_ndp.append(sum(val) / len(val))

        dry_ndp = np.array(dry_ndp)
        new_dry_ndp = np.zeros(_length)
        new_dry_ndp[:dry_ndp.shape[0]] = dry_ndp

        # fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150, constrained_layout=True)
        # widths = np.diff(dp)
        # widths = np.append(widths, widths[-1])
        # ax.bar(dp, ndp, width=widths, alpha=0.3)
        # ax.bar(dp, new_dry_ndp, width=widths, color='g', alpha=0.3)
        # plt.semilogx()
        # ax.config(dp, ndp, ls='solid', color='b', lw=2)
        # ax.config(dp[:np.size(dry_ndp)], dry_ndp, ls='solid', color='r', lw=2)
        # xlim = kwargs.get('xlim') or (11.8, 2500)
        # ylim = kwargs.get('ylim') or (0, 2e5)
        # xlabel = kwargs.get('xlabel') or r'$\bf Diameter\ (nm)$'
        # ylabel = kwargs.get('ylabel') or r'$\bf dN/dlogdp\ (1/Mm)$'
        # ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        # plt.show()

        out_dis['dry_dist'].append(new_dry_ndp)

    dry_PNSD = pd.DataFrame(out_dis['dry_dist']).set_index(_index).set_axis(dp, axis=1).reindex(index)

    return dry_PNSD


def resloved_gRH(length, gRH=1.31, uniform=True):
    if uniform:
        arr = np.array([gRH] * length)

    else:
        lognorm_dist = lambda x, geoMean, geoStd: (gRH / (np.log10(geoStd) * np.sqrt(2 * np.pi))) * np.exp(-(x - np.log10(geoMean))**2 / (2 * np.log10(geoStd)**2))
        abc = lognorm_dist(np.log10(dp/1000), 0.5, 2.0)
        arr = np.where(abc < 1, 1, abc)

    return arr


def score():
    with open(PATH_MAIN / 'PESD_dry.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PESD_dry = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(Path("C:/Users/alex/PycharmProjects/DataPlot/Data") / 'All_data.csv', 'r', encoding='utf-8', errors='ignore') as f:
        Measurement = read_csv(f, parse_dates=['Time'], low_memory=False).set_index('Time')[['Extinction', 'gRH']]

    df = concat([Measurement, PESD_dry], axis=1)

    return 'R'


if __name__ == '__main__':
    dry_PNSD_process(reset=True)
    # score()


