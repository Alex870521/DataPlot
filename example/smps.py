from DataPlot import *


if __name__ == '__main__':
    # import data store in DataPlot/data
    PNSD = DataReader('PNSD_dNdlogdp.csv')
    PSSD = DataReader('PSSD_dSdlogdp.csv')
    PVSD = DataReader('PVSD_dVdlogdp.csv')

    # plot
    for data, unit in zip([PNSD, PSSD, PVSD], ['Number', 'Surface', 'Volume']):
        # plot.distribution.heatmap(data, unit=unit)
        plot.distribution.heatmap_tms(data, unit=unit, freq='60d')

    # Classify the data
    # PNSD_state_class, _ = DataClassifier(df=PNSD, by='State', statistic='Table')
    # plot.distribution.plot_dist(PNSD_state_class, _, unit='Number', additional='error')

    # PNSE_ext_class, _ = DataClassifier(df=PNSD, by='Extinction', statistic='Table', qcut=20)
    # plot.distribution.three_dimension(PNSE_ext_class, unit='Number')

    # plot.distribution.curve_fitting(np.array(PNSE_ext_class.columns, dtype=float), PNSE_ext_class.iloc[0, :], mode=3, unit='Number')
