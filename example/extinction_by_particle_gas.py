from DataPlot import *

if __name__ == '__main__':
    # PG : sum of ext by particle and gas
    df = load_default_data()

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
