import sys

sys.path.extend(['/Users/chanchihyu/PycharmProjects/DataPlot'])

from DataPlot import *

dataset = '/Users/chanchihyu/data/2020能見度計畫/data/All_data.csv'


def use_scatter():
    # example of using plot.scatter
    df = DataBase(dataset)

    df = df[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna()
    plot.scatter(df, x='PM25', y='Vis_LPV', c='VC', s='RH', cmap='YlGnBu')


def use_regression():
    # example of using plot.linear_regression
    df = DataBase(dataset)

    plot.linear_regression(df, x='PM25', y='Extinction')

    plot.linear_regression(df, x='PM25', y=['Extinction', 'Scattering', 'Absorption'])

    plot.multiple_linear_regression(df, x=['AS', 'AN', 'OM', 'EC', 'SS', 'Soil'], y=['Extinction'])

    plot.multiple_linear_regression(df, x=['NO', 'NO2', 'CO', 'PM1'], y=['PM25'])


def use_CBPF_windrose():
    # example of using plot.meteorology
    df = DataBase(dataset)

    plot.meteorology.wind_rose(df, 'WS', 'WD', typ='bar')
    plot.meteorology.wind_rose(df, 'WS', 'WD', 'PM25', typ='scatter')

    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25')
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[0, 25])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[25, 50])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[50, 75])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[75, 100])


def use_SMPS():
    # example of using plot.distribution
    PNSD = DataReader('PNSD_dNdlogdp.csv')
    PSSD = DataReader('PSSD_dSdlogdp.csv')
    PVSD = DataReader('PVSD_dVdlogdp.csv')

    # plot
    for data, unit in zip([PNSD, PSSD, PVSD], ['Number', 'Surface', 'Volume']):
        plot.distribution.heatmap(data, unit=unit)
        plot.distribution.heatmap_tms(data, unit=unit, freq='60d')

    # Classify the data
    # PNSD_state_class, _ = DataClassifier(df=PNSD, by='State', statistic='Table')
    # plot.distribution.plot_dist(PNSD_state_class, _, unit='Number', additional='error')

    # PNSE_ext_class, _ = DataClassifier(df=PNSD, by='Extinction', statistic='Table', qcut=20)
    # plot.distribution.three_dimension(PNSE_ext_class, unit='Number')

    # plot.distribution.curve_fitting(np.array(PNSE_ext_class.columns, dtype=float), PNSE_ext_class.iloc[0, :], mode=3, unit='Number')


if __name__ == '__main__':
    use_SMPS()
