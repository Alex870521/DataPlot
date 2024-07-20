from DataPlot import *

if __name__ == '__main__':
    df = DataBase('/Users/chanchihyu/data/2020能見度計畫/data/All_data.csv').copy()

    plot.meteorology.wind_rose(df, 'WS', 'WD', typ='bar')
    plot.meteorology.wind_rose(df, 'WS', 'WD', 'PM25', typ='scatter')

    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25')
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[0, 25])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[25, 50])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[50, 75])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[75, 100])
