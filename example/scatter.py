from DataPlot import *


if __name__ == '__main__':
    # example of using plot.scatter
    df = load_default_data()

    df = df[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna()
    plot.scatter(df, x='PM25', y='Vis_LPV', c='VC', s='RH', cmap='YlGnBu')
