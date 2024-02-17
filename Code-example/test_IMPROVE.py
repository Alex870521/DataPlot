from DataPlot.plot import *
from DataPlot.process import *

print(dir())

df = DataBase
# fRH_plot()
# ammonium_rich(df)
wind_rose(df['WS'], df['WD'])

