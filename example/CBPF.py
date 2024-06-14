from DataPlot import *

if __name__ == '__main__':
    df = DataBase().copy()
    df1 = df[['WS', 'WD', 'PM25', 'NO2', 'O3', 'SO2']]

    plot.meteorology.wind_rose(df1, 'WS', 'WD', typ='bar')
    plot.meteorology.wind_rose(df1, 'WS', 'WD', 'PM25', typ='scatter')

    plot.meteorology.CBPF(df1, 'WS', 'WD', 'PM25')
    plot.meteorology.CBPF(df1, 'WS', 'WD', 'PM25', percentile=[0, 25])
    plot.meteorology.CBPF(df1, 'WS', 'WD', 'PM25', percentile=[25, 50])
    plot.meteorology.CBPF(df1, 'WS', 'WD', 'PM25', percentile=[50, 75])
    plot.meteorology.CBPF(df1, 'WS', 'WD', 'PM25', percentile=[75, 100])
