from DataPlot import *


def verify_scat_plot():
    plot.templates.linear_regression(DataBase, x='Extinction', y=['Bext_internal', 'Bext_external'], xlim=[0, 300],
                                     ylim=[0, 600])
    plot.templates.linear_regression(DataBase, x='Scattering', y=['Bsca_internal', 'Bsca_external'], xlim=[0, 300],
                                     ylim=[0, 600])
    plot.templates.linear_regression(DataBase, x='Absorption', y=['Babs_internal', 'Babs_external'], xlim=[0, 100],
                                     ylim=[0, 200])


def extinction_sensitivity():
    plot.templates.scatter(DataBase, x='Extinction', y='Bext_Fixed_PNSD', xlim=[0, 600], ylim=[0, 600],
                           title='Fixed PNSD',
                           regression=True, diagonal=True)
    plot.templates.scatter(DataBase, x='Extinction', y='Bext_Fixed_RI', xlim=[0, 600], ylim=[0, 600], title='Fixed RI',
                           regression=True, diagonal=True)


if __name__ == '__main__':
    extinction_sensitivity()
