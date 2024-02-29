from pandas import concat
from DataPlot.process import DataBase, DataReader, Classifier
from DataPlot.plot.templates import scatter, linear_regression, multiple_linear_regression, donuts_ext


def residual_PM(_df):
    _df['residual_PM'] = _df['PM25'] - _df['AS'] - _df['AN'] - _df['OM'] - _df['SS'] - _df['EC']

    return _df[['residual_PM', 'Ti', 'Fe', 'Si']]


def residual_ext(_df):
    _df['residual_ext'] = _df['total_ext_dry'] - _df['AS_ext_dry'] - _df['AN_ext_dry'] - _df['Soil_ext_dry'] - _df[
        'SS_ext_dry']

    return _df[['residual_ext', 'POC', 'SOC']]


def MLR_IMPROVE():
    df = DataBase

    species = ['Extinction', 'Scattering', 'Absorption', 'total_ext_dry', 'AS_ext_dry', 'AN_ext_dry',
               'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry',
               'AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'OM']

    df = df[species].dropna().copy()

    # multiple_linear_regression(df, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS'], y='Scattering', add_constant=True)
    # multiple_linear_regression(df, x=['POC', 'SOC', 'EC'], y='Absorption', add_constant=True)
    # multiple_linear_regression(df, x=['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC'], y='Extinction', add_constant=True)

    multiplier = [2.74, 4.41, 11.5, 7.34, 12.27]
    df['Localized'] = df[['AS', 'AN', 'POC', 'SOC', 'EC']].mul(multiplier).sum(axis=1)
    modify_IMPROVE = DataReader('modify_IMPROVE.csv')['total_ext_dry'].rename('Modified')
    revised_IMPROVE = DataReader('revised_IMPROVE.csv')['total_ext_dry'].rename('Revised')

    df = concat([df, revised_IMPROVE, modify_IMPROVE], axis=1)

    n_df = df[['AS', 'AN', 'POC', 'SOC', 'EC']].mul(multiplier)
    new_df = concat([df['Extinction'], n_df], axis=1)
    new_dic = Classifier(new_df, 'State')

    ext_dry_dict = {state: [new_dic[state][specie].mean() for specie in ['AS', 'AN', 'POC', 'SOC', 'EC']]
                    for state in ['Total', 'Clean', 'Transition', 'Event']}

    # plot
    linear_regression(df, x='Extinction', y=['Revised', 'Modified', 'Localized'], xlim=[0, 400], ylim=[0, 400],
                      regression=True, diagonal=True)
    donuts_ext(ext_dry_dict, labels=['AS', 'AN', 'POC', 'SOC', 'EC'])


if __name__ == '__main__':
    MLR_IMPROVE()
