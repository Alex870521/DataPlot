from DataPlot import *

# linaer_regression
plot.templates.linear_regression(DataBase, x='PM25', y='Extinction')

# multiple_linear_regression
plot.templates.multiple_linear_regression(DataBase, ['AS', 'AN', 'OM', 'BC', 'SS', 'Soil'], y=['Extinction'])
