from DataPlot import *
import pandas as pd


if __name__ == '__main__':
    PNSD = DataReader('PNSD_dNdlogdp.csv')
    PSSD = DataReader('PSSD_dSdlogdp.csv')
    PVSD = DataReader('PVSD_dVdlogdp.csv')
    PESD = DataReader('PESD_dextdlogdp_internal.csv')
    Data = DataBase

    # Season timeseries
    for season, (st_tm_, fn_tm_) in DataClassifier.Seasons.items():
        st_tm, fn_tm = pd.Timestamp(st_tm_), pd.Timestamp(fn_tm_)

        df = Data.loc[st_tm:fn_tm].copy()

        PNSD_data = PNSD.loc[st_tm:fn_tm]
        PSSD_data = PSSD.loc[st_tm:fn_tm]

        # 數據平滑
        # df = df.rolling(6).mean(numeric_only=True)

        plot.timeseries(df)
        plot.heatmap_tms(PNSD_data)
