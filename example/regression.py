from DataPlot import *

# linaer_regression
plot.linear_regression(DataBase, x='PM25', y='Extinction')

plot.linear_regression(DataBase, x='PM25', y=['Extinction', 'Scattering', 'Absorption'])

# multiple_linear_regression
plot.multiple_linear_regression(DataBase, x=['AS', 'AN', 'OM', 'EC', 'SS', 'Soil'], y=['Extinction'])
