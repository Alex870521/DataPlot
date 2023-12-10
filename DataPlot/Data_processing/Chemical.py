import numpy as np
import pandas as pd
from pandas import read_csv, concat
from pathlib import Path
from DataPlot.Data_processing.decorator.csv_decorator import save_to_csv

PATH_MAIN = Path(__file__).parents[2] / 'Data-example'

# Note
# df['ALWC'] 不要加到 df_volume裡面


def mass(_df):  # Series like
    Ammonium, Sulfate, Nitrate, OC, Soil, SS, EC, PM25 = _df
    status = (Ammonium / 18) / (2 * (Sulfate / 96) + (Nitrate / 62))

    if status >= 1:
        _df['NH4_status'] = 'Enough'
        _df['AS'] = (1.375 * Sulfate)
        _df['AN'] = (1.29 * Nitrate)

    if status < 1:
        _df['NH4_status'] = 'Deficiency'
        mol_A = Ammonium / 18
        mol_S = Sulfate / 96
        mol_N = Nitrate / 62
        residual = mol_A - 2 * mol_S

        if residual > 0:
            if residual <= mol_N:
                _df['AS'] = (1.375 * Sulfate)
                _df['AN'] = (residual * 80)

            if residual > mol_N:
                _df['AS'] = (1.375 * Sulfate)
                _df['AN'] = (mol_N * 80)

        if residual <= 0:
            if mol_A <= 2 * mol_S:
                _df['AS'] = (mol_A / 2 * 132)
                _df['AN'] = 0

            if mol_A > 2 * mol_S:
                _df['AS'] = (mol_S * 132)
                _df['AN'] = 0

    _df['OM'] = (1.8 * OC)
    _df['Soil'] = (28.57 * Soil)
    _df['SS'] = (2.54 * SS)
    _df['EC'] = EC
    _df['SIA'] = _df['AS'] + _df['AN']
    _df['total_mass'] = _df[['AS', 'AN', 'OM', 'Soil', 'SS', 'EC']].sum()
    species_lst = ['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'SIA', 'unknown_mass']

    if PM25 >= _df['total_mass']:
        _df['unknown_mass'] = PM25 - _df['total_mass']
        for _species, _val in _df[species_lst].items():
            _df[f'{_species}_ratio'] = _val / PM25

    else:
        _df['unknown_mass'] = 0
        for _species, _val in _df[species_lst].items():
            _df[f'{_species}_ratio'] = _val / _df['total_mass']

    return _df['NH4_status':]


def volume(_df):
    _df['AS_volume']    = (_df['AS'] / 1.76)
    _df['AN_volume']    = (_df['AN'] / 1.73)
    _df['OM_volume']    = (_df['OM'] / 1.4)
    _df['Soil_volume']  = (_df['Soil'] / 2.6)
    _df['SS_volume']    = (_df['SS'] / 2.16)
    _df['EC_volume']    = (_df['EC'] / 1.5)
    _df['total_volume'] = sum(_df['AS_volume':'EC_volume'])

    for _species, _val in _df['AS_volume':'EC_volume'].items():
        _df[f'{_species}_ratio'] = _val / _df['total_volume']

    _df['density'] = _df['total_mass'] / _df['total_volume']
    return _df['AS_volume':]


def volume_average_mixing(_df):
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

    # 檢查_df['ALWC']是否缺失 -> 有值才計算ambient的折射率
    if pd.notna(_df['ALWC']):
        v_dry = _df['total_volume']
        v_wet = _df['total_volume'] + _df['ALWC']

        multiplier = v_dry / v_wet
        _df['ALWC_volume_ratio'] = (1-multiplier)

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


@save_to_csv(PATH_MAIN / 'Level2' / 'chemical.csv')
def chemical_process(reset=False, filename=None):
    if filename.exists() & (~reset):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'level1' / 'EPB.csv', 'r', encoding='utf-8', errors='ignore') as f:
        minion = read_csv(f, parse_dates=['Time'], na_values=['-', 'E', 'F']).set_index('Time')

    with open(PATH_MAIN / 'level1' / 'IMPACT.csv', 'r', encoding='utf-8', errors='ignore') as f:
        impact = read_csv(f, parse_dates=['Time']).set_index('Time')

    df = concat([minion, impact], axis=1)

    df_mass = df[['NH4+', 'SO42-', 'NO3-', 'O_OC', 'Fe', 'Na+', 'O_EC', 'PM25']].dropna().copy().apply(mass, axis=1)
    df_volume = df_mass[['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'total_mass']].dropna().copy().apply(volume, axis=1)
    df_volume['ALWC'] = df['ALWC']
    df_vam = df_volume.copy().apply(volume_average_mixing, axis=1)

    return concat([df_mass, df_volume.drop(['ALWC'], axis=1), df_vam], axis=1).reindex(df.index.copy())


if __name__ == '__main__':
    df = chemical_process(reset=True)
