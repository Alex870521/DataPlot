from DataPlot import *


if __name__ == '__main__':
    # FIXME:
    # example of using plot.scatter
    df = DataBase[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna()
    ax = plot.scatter(DataBase, x='PM25', y='Vis_LPV', c='VC', s='RH', cmap='YlGnBu')
