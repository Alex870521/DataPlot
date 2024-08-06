import sys

sys.path.extend(['/Users/chanchihyu/PycharmProjects/DataPlot'])

from DataPlot import *

dataset = '/Users/chanchihyu/NTU/2020能見度計畫/data/All_data.csv'


def use_scatter():
    # example of using plot.scatter
    df = DataBase(dataset)

    df = df[['Vis_LPV', 'PM25', 'RH', 'VC']].dropna()
    plot.scatter(df, x='PM25', y='Vis_LPV', c='VC', s='RH', cmap='YlGnBu')


def use_regression():
    # example of using plot.linear_regression
    df = DataBase(dataset)

    plot.linear_regression(df, x='PM25', y='Extinction')
    plot.linear_regression(df, x='PM25', y=['Extinction', 'Scattering', 'Absorption'])
    plot.multiple_linear_regression(df, x=['AS', 'AN', 'OM', 'EC', 'SS', 'Soil'], y=['Extinction'])
    plot.multiple_linear_regression(df, x=['NO', 'NO2', 'CO', 'PM1'], y=['PM25'])


def use_CBPF_windrose():
    # example of using plot.meteorology
    df = DataBase(dataset)

    plot.meteorology.wind_rose(df, 'WS', 'WD', typ='bar')
    plot.meteorology.wind_rose(df, 'WS', 'WD', 'PM25', typ='scatter')

    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25')
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[0, 25])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[25, 50])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[50, 75])
    plot.meteorology.CBPF(df, 'WS', 'WD', 'PM25', percentile=[75, 100])


def use_SMPS():
    # example of using plot.distribution
    PNSD = DataReader('PNSD_dNdlogdp.csv')
    PSSD = DataReader('PSSD_dSdlogdp.csv')
    PVSD = DataReader('PVSD_dVdlogdp.csv')

    # plot
    for data, unit in zip([PNSD, PSSD, PVSD], ['Number', 'Surface', 'Volume']):
        plot.distribution.heatmap(data, unit=unit)
        plot.distribution.heatmap_tms(data, unit=unit, freq='60d')

    # Classify the data
    # PNSD_state_class, _ = DataClassifier(df=PNSD, by='State', statistic='Table')
    # plot.distribution.plot_dist(PNSD_state_class, _, unit='Number', additional='error')

    # PNSE_ext_class, _ = DataClassifier(df=PNSD, by='Extinction', statistic='Table', qcut=20)
    # plot.distribution.three_dimension(PNSE_ext_class, unit='Number')

    # plot.distribution.curve_fitting(np.array(PNSE_ext_class.columns, dtype=float), PNSE_ext_class.iloc[0, :], mode=3, unit='Number')


def use_linear_regression_and_scatter_to_verify():
    # example of using plot.linear_regression and plot.scatter
    df = DataBase(dataset)

    plot.linear_regression(df, x='Extinction', y=['Bext_internal', 'Bext_external'], xlim=[0, 300], ylim=[0, 600])
    plot.linear_regression(df, x='Scattering', y=['Bsca_internal', 'Bsca_external'], xlim=[0, 300], ylim=[0, 600])
    plot.linear_regression(df, x='Absorption', y=['Babs_internal', 'Babs_external'], xlim=[0, 100], ylim=[0, 200])

    plot.scatter(df, x='Extinction', y='Bext_Fixed_PNSD', xlim=[0, 600], ylim=[0, 600], title='Fixed PNSD',
                 regression=True, diagonal=True)
    plot.scatter(df, x='Extinction', y='Bext_Fixed_RI', xlim=[0, 600], ylim=[0, 600], title='Fixed RI',
                 regression=True, diagonal=True)


def use_extinction_by_particle_gas():
    # example of using plot.bar and plot.pie
    df = DataBase(dataset)

    ser_grp_sta, ser_grp_sta_std = DataClassifier(df, by='State')
    ext_particle_gas = ser_grp_sta.loc[:, ['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas']]

    plot.bar(data_set=ext_particle_gas, data_std=None,
             labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
             unit='Extinction',
             style="stacked",
             colors=plot.Color.paired)

    plot.pie(data_set=ext_particle_gas,
             labels=[rf'$b_{{sp}}$', rf'$b_{{ap}}$', rf'$b_{{sg}}$', rf'$b_{{ag}}$'],
             unit='Extinction',
             style='donut',
             colors=plot.Color.paired)


def use_timeseries():
    # example of using plot.timeseries
    df = DataBase(dataset)

    plot.timeseries(df,
                    y=['Extinction', 'Scattering', 'Absorption'],
                    y2=['PBLH'],
                    c=[None, None, None, 'VC'],
                    style=['line', 'line', 'line', 'scatter'],
                    times=('2020-10-01', '2020-11-30'), ylim=[0, None], ylim2=[0, None], rolling=50,
                    inset_kws2=dict(bbox_to_anchor=(1.12, 0, 1.2, 1)))

    plot.timeseries(df, y='WS', c='WD', style='scatter', times=('2020-10-01', '2020-11-30'),
                    scatter_kws=dict(cmap='hsv'), cbar_kws=dict(ticks=[0, 90, 180, 270, 360]),
                    ylim=[0, None])

    plot.timeseries_template(df.loc['2020-09-01':'2020-12-31'])


if __name__ == '__main__':
    # use_SMPS()
    # use_CBPF_windrose()
    # use_extinction_by_particle_gas()
    use_timeseries()
