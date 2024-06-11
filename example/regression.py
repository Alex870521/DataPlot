from DataPlot import *

if __name__ == '__main__':
    df = load_default_data()

    # linaer_regression
    # plot.linear_regression(df, x='PM25', y='Extinction')

    # plot.linear_regression(df, x='PM25', y=['Extinction', 'Scattering', 'Absorption'])

    # multiple_linear_regression
    # plot.multiple_linear_regression(df, x=['AS', 'AN', 'OM', 'EC', 'SS', 'Soil'], y=['Extinction'])
    plot.multiple_linear_regression(df, x=['NO', 'NO2', 'CO', 'PM1'], y=['PM25'])
