from DataPlot import *
from heatmap import heatmap, corrplot
import seaborn as sns

df = DataBase

# plot.fRH_plot()
# plot.ammonium_rich(df)
# plot.wind_rose(df['WS'], df['WD'])
# plot.MLR_IMPROVE()
species = ['Extinction', 'Scattering', 'Absorption', 'PM1', 'PM25', 'PM10', 'NO', 'NO2', 'NOx', 'O3', 'Benzene', 'Toluene']
corr = df[species].dropna().corr()

ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)