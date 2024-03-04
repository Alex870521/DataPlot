from DataPlot.plot import scatter, linear_regression
from DataPlot.process import DataBase, DataReader, SizeDist


def verify_scat_plot():
    linear_regression(DataBase, x='Extinction', y=['Bext_internal', 'Bext_external'], xlim=[0, 300], ylim=[0, 600])
    linear_regression(DataBase, x='Scattering', y=['Bsca_internal', 'Bsca_external'], xlim=[0, 300], ylim=[0, 600])
    linear_regression(DataBase, x='Absorption', y=['Babs_internal', 'Babs_external'], xlim=[0, 100], ylim=[0, 200])


def extinction_sensitivity():
    scatter(DataBase, x='Extinction', y='Bext_Fixed_PNSD', xlim=[0, 600], ylim=[0, 600], title='Fixed PNSD', regression=True, diagonal=True)
    scatter(DataBase, x='Extinction', y='Bext_Fixed_RI', xlim=[0, 600], ylim=[0, 600], title='Fixed RI', regression=True, diagonal=True)


if __name__ == '__main__':
    extinction_sensitivity()
