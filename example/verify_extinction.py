from DataPlot import *


def verify_scat_plot():
    df = load_default_data()

    plot.linear_regression(df, x='Extinction', y=['Bext_internal', 'Bext_external'], xlim=[0, 300], ylim=[0, 600])
    plot.linear_regression(df, x='Scattering', y=['Bsca_internal', 'Bsca_external'], xlim=[0, 300], ylim=[0, 600])
    plot.linear_regression(df, x='Absorption', y=['Babs_internal', 'Babs_external'], xlim=[0, 100], ylim=[0, 200])


def extinction_sensitivity():
    df = load_default_data()

    plot.scatter(df, x='Extinction', y='Bext_Fixed_PNSD', xlim=[0, 600], ylim=[0, 600], title='Fixed PNSD',
                 regression=True, diagonal=True)
    plot.scatter(df, x='Extinction', y='Bext_Fixed_RI', xlim=[0, 600], ylim=[0, 600], title='Fixed RI',
                 regression=True, diagonal=True)


if __name__ == '__main__':
    extinction_sensitivity()
