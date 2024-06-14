from DataPlot import *

if __name__ == '__main__':
    df = DataBase().copy()
    df1 = df[['WS', 'WD', 'PM25', 'NO2', 'O3', 'SO2']]

    plot.meteorology.wind_rose(df1, 'WS', 'WD', typ='bar')
    plot.meteorology.wind_rose(df1, 'WS', 'WD', 'PM25', typ='scatter')

    plot.meteorology.wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf')
    plot.meteorology.wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf', percentile=[0, 25])
    plot.meteorology.wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf', percentile=[25, 50])
    plot.meteorology.wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf', percentile=[50, 75])
    plot.meteorology.wind_rose(df1, 'WS', 'WD', 'PM25', typ='cbpf', percentile=[75, 100])
