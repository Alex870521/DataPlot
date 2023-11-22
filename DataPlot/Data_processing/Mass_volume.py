import numpy as np
import pandas as pd
from pandas import read_csv, concat
from pathlib import Path
from DataPlot.Data_processing.csv_decorator import save_to_csv

PATH_MAIN = Path(__file__).parent.parent.parent / 'Data'


def mass(_df):  # Series like
    Ammonium, Sulfate, Nitrate, OC, Soil, SS, EC, PM25 = _df
    status = (Ammonium / 18) / (2 * (Sulfate / 96) + (Nitrate / 62))

    if status >= 1:
        _df['NH4_status'] = 'Enough'
        _df['AS_mass'] = (1.375 * Sulfate)
        _df['AN_mass'] = (1.29 * Nitrate)
        _df['OM_mass'] = (1.8 * OC)
        _df['Soil_mass'] = (28.57 * Soil)
        _df['SS_mass'] = (2.54 * SS)
        _df['EC_mass'] = EC
        _df['total_mass'] = sum(_df['AS_mass':'EC_mass'])
        _df['SIA_mass'] = _df['AS_mass'] + _df['AN_mass']

    if status < 1:
        _df['NH4_status'] = 'Deficiency'
        mol_A = Ammonium / 18
        mol_S = Sulfate / 96
        mol_N = Nitrate / 62
        residual = mol_A - 2 * mol_S

        if residual > 0:
            if residual <= mol_N:
                _df['AS_mass'] = (1.375 * Sulfate)
                _df['AN_mass'] = (residual * 80)

            if residual > mol_N:
                _df['AS_mass'] = (1.375 * Sulfate)
                _df['AN_mass'] = (mol_N * 80)

        if residual <= 0:
            if mol_A <= 2 * mol_S:
                _df['AS_mass'] = (mol_A / 2 * 132)
                _df['AN_mass'] = 0

            if mol_A > 2 * mol_S:
                _df['AS_mass'] = (mol_S * 132)
                _df['AN_mass'] = 0

        _df['OM_mass'] = (1.8 * OC)
        _df['Soil_mass'] = (28.57 * Soil)
        _df['SS_mass'] = (2.54 * SS)
        _df['EC_mass'] = EC
        _df['SIA_mass'] = _df['AS_mass'] + _df['AN_mass']
        _df['total_mass'] = sum(_df['AS_mass':'EC_mass'])

    if _df['PM25'] >= _df['total_mass']:
        _df['others_mass'] = _df['PM25'] - _df['total_mass']
        for _val, _species in zip(_df['AS_mass':'others_mass'], _df['AS_mass':'others_mass'].index):
            _df[f'{_species}_ratio'] = _val / _df['PM25']

    if _df['PM25'] < _df['total_mass']:
        _df['others_mass'] = 0
        for _val, _species in zip(_df['AS_mass':'others_mass'], _df['AS_mass':'others_mass'].index):
            _df[f'{_species}_ratio'] = _val / _df['total_mass']

    return _df['NH4_status':].drop(labels=['total_mass_ratio'])


def volume(_df):
    _df['AS_volume']    = (_df['AS_mass'] / 1.76)
    _df['AN_volume']    = (_df['AN_mass'] / 1.73)
    _df['OM_volume']    = (_df['OM_mass'] / 1.4)
    _df['Soil_volume']  = (_df['Soil_mass'] / 2.6)
    _df['SS_volume']    = (_df['SS_mass'] / 2.16)
    _df['EC_volume']    = (_df['EC_mass'] / 1.5)
    _df['total_volume'] = sum(_df['AS_volume':'EC_volume'])

    for _val, _species in zip(_df['AS_volume':'EC_volume'].values, _df['AS_volume':'EC_volume'].index):
        _df[f'{_species}_ratio'] = _val / _df['total_volume']

    _df['density'] = _df['total_mass'] / _df['total_volume']
    return _df['AS_volume':]


def VAM(_df):
    # volume_average_mixing
    _df['n_dry'] = (1.53 * _df['AS_volume_ratio'] +
                    1.55 * _df['AN_volume_ratio'] +
                    1.55 * _df['OM_volume_ratio'] +
                    1.56 * _df['Soil_volume_ratio'] +
                    1.54 * _df['SS_volume_ratio'] +
                    1.80 * _df['EC_volume_ratio'])

    _df['k_dry'] = (0.00 * _df['OM_volume_ratio'] +
                    0.01 * _df['Soil_volume_ratio'] +
                    0.54 * _df["EC_volume_ratio"])

    if pd.notna(_df['ALWC_volume']):

        v_dry = _df['total_volume']
        v_wet = _df['total_volume'] + _df['ALWC_volume']

        multiplier = v_dry / v_wet

        _df['ALWC_volume_ratio'] = 1-multiplier

        _df['n_amb'] = (1.53 * _df['AS_volume_ratio'] +
                        1.55 * _df['AN_volume_ratio'] +
                        1.55 * _df['OM_volume_ratio'] +
                        1.56 * _df['Soil_volume_ratio'] +
                        1.54 * _df['SS_volume_ratio'] +
                        1.80 * _df['EC_volume_ratio']) * multiplier + \
                       (1.33 * _df['ALWC_volume_ratio'])

        _df['k_amb'] = (0.00 * _df['OM_volume_ratio'] +
                        0.01 * _df['Soil_volume_ratio'] +
                        0.54 * _df['EC_volume_ratio']) * multiplier

        _df['gRH'] = (v_wet / v_dry) ** (1 / 3)

    _df['kappa_chem'] = np.nan

    return _df['n_dry':]


@save_to_csv(PATH_MAIN / 'Level2' / 'mass_volume_VAM.csv')
def mass_volume_process(reset=False, filename=None):
    if filename.exists() & (~reset):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'level1' / 'EPB.csv', 'r', encoding='utf-8', errors='ignore') as f:
        minion = read_csv(f, parse_dates=['Time'], na_values=['-', 'E', 'F']).set_index('Time')

    with open(PATH_MAIN / 'level1' / 'IMPACT.csv', 'r', encoding='utf-8', errors='ignore') as f:
        impact = read_csv(f, parse_dates=['Time']).set_index('Time')

    df = concat([minion, impact], axis=1)

    _index = df.index.copy()
    df_mass = df[['NH4+', 'SO42-', 'NO3-', 'O_OC', 'Fe', 'Na+', 'O_EC', 'PM25']].dropna().copy().apply(mass, axis=1).reindex(_index)
    df_mass.index.name = 'Time'
    df_mass['ALWC_mass'] = df['ALWC']
    df_mass['ALWC_mass_ratio'] = df_mass['ALWC_mass'] / df_mass['total_mass']

    df_volume = df_mass[['AS_mass', 'AN_mass', 'OM_mass', 'Soil_mass', 'SS_mass', 'EC_mass', 'total_mass']].dropna().copy().apply(volume, axis=1).reindex(_index)
    df_volume.index.name = 'Time'
    df_volume['ALWC_volume'] = df['ALWC']

    # did not dropna for ALWC_volume
    df_VAM = df_volume.copy().apply(VAM, axis=1).reindex(_index)
    df_VAM.index.name = 'Time'

    All_df = concat([df_mass, df_volume, df_VAM], axis=1)

    return All_df


if __name__ == '__main__':
    df = mass_volume_process(reset=True)
