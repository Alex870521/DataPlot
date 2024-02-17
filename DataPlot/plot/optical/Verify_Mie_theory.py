import numpy as np
import math
import pandas as pd
from pathlib import Path
from pandas import read_csv, concat
import pickle
from DataPlot.process import *
from DataPlot.plot import *
import matplotlib.pyplot as plt
import seaborn as sns


PATH_MAIN = Path(__file__).parent.parent / 'Data-Code-example'
PATH_DIST = PATH_MAIN / 'Level2' / 'distribution'

PNSD, All = DataReader('PNSD_dNdlogdp.csv'), DataReader('All_data.csv')

df = concat([All, PNSD], axis=1)

dp = np.array(PNSD.columns, dtype='float')
_length = np.size(dp)
dlogdp = np.array([0.014] * _length)


def verify_scat_plot():
    # fix, ax = scatter_mutiReg(df, x='Extinction', y1='Bext', y2='Bext_external', xlim=[0, 300], ylim=[0, 600], title='', regression=True, diagonal=True)
    # fix, ax = scatter_mutiReg(df, x='Scattering', y1='Bsca', y2='Bsca_external', xlim=[0, 300], ylim=[0, 600], title='', regression=True, diagonal=True)
    # fix, ax = scatter_mutiReg(df, x='Absorption', y1='Babs', y2='Babs_external', xlim=[0, 100], ylim=[0, 200], title='', regression=True, diagonal=True)

    # fix, ax = scatter_mutiReg(df, x='Extinction', y1='Bext_dry', y2='Bext_dry_external', xlim=[0, 300], ylim=[0, 600], title='', regression=True, diagonal=True)
    # fix, ax = scatter_mutiReg(df, x='Scattering', y1='Bsca_dry', y2='Bsca_dry_external', xlim=[0, 300], ylim=[0, 600], title='', regression=True, diagonal=True)
    # fix, ax = scatter_mutiReg(df, x='Absorption', y1='Babs_dry', y2='Babs_dry_external', xlim=[0, 100], ylim=[0, 200], title='', regression=True, diagonal=True)

    # scatter(df, x='Extinction', y='Bext_dry', c='gRH', c_range=[1, 2], xlim=[0, 600], ylim=[0, 600], regression=True, diagonal=True)
    # scatter(df, x='Scattering', y='Bsca_dry', c='gRH', c_range=[1, 2], xlim=[0, 500], ylim=[0, 500], regression=True, diagonal=True)
    # scatter(df, x='Absorption', y='Babs_dry', c='gRH', c_range=[1, 2], xlim=[0, 200], ylim=[0, 200], regression=True, diagonal=True)

    scatter(df, x='Extinction', y='Bext_internal', c='RH', c_range=[50, 100], xlim=[0, 700], ylim=[0, 700], regression=True, diagonal=True)
    scatter(df, x='Extinction', y='Bext_external', c='RH', c_range=[50, 100], xlim=[0, 700], ylim=[0, 700], regression=True, diagonal=True)

    # scatter(df, x='Scattering', y='Bsca', c='RH', c_range=[50, 100], xlim=[0, 600], ylim=[0, 600], regression=True, diagonal=True)
    # scatter(df, x='Scattering', y='Bsca_external', c='RH', c_range=[50, 100], xlim=[0, 600], ylim=[0, 600], regression=True, diagonal=True)

    # scatter(df, x='Absorption', y='Babs', c='RH', c_range=[50, 100], xlim=[0, 200], ylim=[0, 200], regression=True, diagonal=True)
    # scatter(df, x='Absorption', y='Babs_external', c='RH', c_range=[50, 100], xlim=[0, 200], ylim=[0, 200], regression=True, diagonal=True)


if __name__ == '__main__':
    # df1 = df[['Extinction', 'Bext']].dropna()
    # df2 = pd.DataFrame('Internal', index=df1.index, columns=['method'])
    # df_merged = pd.concat([df1, df2], axis=1)
    #
    # df1_ = df[['Extinction', 'Bext_external']].dropna()
    # df2_ = pd.DataFrame('External', index=df1_.index, columns=['method'])
    # df_merged_ = pd.concat([df1_, df2_], axis=1)
    # df_merged_.rename(columns={'Bext_external': 'Bext'}, inplace=True)
    #
    # df = pd.merge(df_merged, df_merged_, on=['Time', 'Extinction', 'Bext', 'method'], how='outer')
    # df.reset_index(drop=True, inplace=True)
    sns.jointplot(data=df, x="Extinction", y="Bext_internal")

    # verify_scat_plot()

    # scatter(df, x='PM25', y='total_mass', xlim=[0, 100], ylim=[0, 100], regression=True, diagonal=True)
    # scatter(df, x='PM1/PM25', y='MEE')
    # scatter(df, x='PM25', y='Extinction', c='PM1/PM25', c_range=[0, 1], regression=True, )
    # scatter(df, x='PM25', y='Scattering', c='PM1/PM25', c_range=[0, 1], regression=True, )
    # scatter(df, x='PM25', y='Absorption', c='PM1/PM25', c_range=[0, 1], regression=True, )
    # scatter(df, x='PM1', y='Extinction', c='PM1/PM25', c_range=[0, 1], regression=True, )
    # scatter(df, x='Extinction', y='O3', c='PM1', y_range=[0, 15], c_range=[0, 1])

