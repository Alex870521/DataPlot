from pandas import concat
from DataPlot.process import *
from DataPlot.plot import *


def MLR_IMPROVE():
    species = ['Extinction', 'Scattering', 'Absorption', 'total_ext_dry', 'AS_ext_dry', 'AN_ext_dry',
               'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry',
               'AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'OM']

    df = DataBase[species].dropna().copy()

    # multiple_linear_regression(df, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS'], y='Scattering', add_constant=True)
    # multiple_linear_regression(df, x=['POC', 'SOC', 'EC'], y='Absorption', add_constant=True)
    # multiple_linear_regression(df, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], y='Extinction', add_constant=True)

    multiplier = [2.675, 4.707, 11.6, 7.272, 0, 0.131, 10.638]
    df['Localized'] = df[['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC']].mul(multiplier).sum(axis=1)
    modify_IMPROVE = DataReader('modified_IMPROVE.csv')['total_ext_dry'].rename('Modified')
    revised_IMPROVE = DataReader('revised_IMPROVE.csv')['total_ext_dry'].rename('Revised')

    df = concat([df, revised_IMPROVE, modify_IMPROVE], axis=1)

    n_df = df[['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC']].mul(multiplier)
    mean, std = DataClassifier(n_df, 'State', statistic='Table')

    # plot
    linear_regression(df, x='Extinction', y=['Revised', 'Modified', 'Localized'], xlim=[0, 400], ylim=[0, 400],
                      regression=True, diagonal=True)
    Pie.donuts(mean, labels=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], unit='Extinction', colors=Color.colors3)


if __name__ == '__main__':
    MLR_IMPROVE()
