from DataPlot import *


if __name__ == '__main__':
    # import data store in DataPlot/data
    PNSD = DataReader('PNSD_dNdlogdp.csv')

    # plot
    # plot.distribution.heatmap_tms(PNSD, freq='60d')

    # Classify the data
    PNSD_state_class, _ = DataClassifier(df=PNSD, by='State', statistic='Table')
    plot.distribution.plot_dist(PNSD_state_class, _, unit='Number', additional='std')

    # PNSE_ext_class, _ = DataClassifier(df=PNSD, by='Extinction', statistic='Table', qcut=20)
    # plot.distribution.three_dimension(PNSE_ext_class, unit='Number')


