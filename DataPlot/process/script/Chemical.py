from pathlib import Path

import numpy as np
from pandas import read_csv, concat, notna, DataFrame

from DataPlot.process.core import *


class ChemicalProcessor(DataProcessor):
    """
    A class for process chemical data.

    Parameters:
    -----------
    reset : bool, optional
        If True, resets the process. Default is False.
    filename : str, optional
        The name of the file to process. Default is None.

    Methods:
    --------
    mass(_df):
        Calculate mass-related parameters.

    volume(_df):
        Calculate volume-related parameters.

    volume_average_mixing(_df):
        Calculate volume average mixing parameters.

    process_data():
        Process data and save the result.

    Attributes:
    -----------
    DEFAULT_PATH : Path
        The default path for data files.

    Examples:
    ---------

    """

    def __init__(self):
        super().__init__()
        self.file_path = self.DEFAULT_PATH / 'Level2'

    @staticmethod
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

    @staticmethod
    def volume(_df):
        _df['AS_volume'] = (_df['AS'] / 1.76)
        _df['AN_volume'] = (_df['AN'] / 1.73)
        _df['OM_volume'] = (_df['OM'] / 1.4)
        _df['Soil_volume'] = (_df['Soil'] / 2.6)
        _df['SS_volume'] = (_df['SS'] / 2.16)
        _df['EC_volume'] = (_df['EC'] / 1.5)
        _df['total_volume'] = sum(_df['AS_volume':'EC_volume'])

        for _species, _val in _df['AS_volume':'EC_volume'].items():
            _df[f'{_species}_ratio'] = _val / _df['total_volume']

        _df['density'] = _df['total_mass'] / _df['total_volume']
        return _df['AS_volume':]

    @staticmethod
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
        if notna(_df['ALWC']):
            v_dry = _df['total_volume']
            v_wet = _df['total_volume'] + _df['ALWC']

            multiplier = v_dry / v_wet
            _df['ALWC_volume_ratio'] = (1 - multiplier)

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
        _df['kappa_vam'] = np.nan

    @staticmethod
    def kappa(_df, diameter=0.5):
        water_surface_tension, water_Mw, water_density, universal_gas_constant = 0.072, 18, 1, 8.314  # J/mole*K

        A = 4 * (water_surface_tension * water_Mw) / (water_density * universal_gas_constant * (_df['AT'] + 273))
        power = A / diameter
        a_w = (_df['RH'] / 100) * (np.exp(-power))

        return (_df['gRH'] ** 3 - 1) * (1 - a_w) / a_w

    @staticmethod
    def ISORROPIA():
        pass

    def process_data(self, reset: bool = False, save_filename: str | Path = 'chemical.csv'):
        file = self.file_path / save_filename
        if file.exists() and not reset:
            return read_csv(file, parse_dates=['Time']).set_index('Time')
        else:
            data_files = ['EPB.csv', 'IMPACT.csv']
            df = concat([DataReader(file) for file in data_files], axis=1)

            df_mass = df[['NH4+', 'SO42-', 'NO3-', 'O_OC', 'Fe', 'Na+', 'O_EC', 'PM25']].dropna().apply(self.mass,
                                                                                                        axis=1)
            df_volume = df_mass[['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'total_mass']].dropna().apply(self.volume,
                                                                                                     axis=1)
            df_volume['ALWC'] = df['ALWC']
            df_vam = df_volume.copy().apply(self.volume_average_mixing, axis=1)

            _df = concat([df_mass, df_volume.drop(['ALWC'], axis=1), df_vam], axis=1).reindex(df.index.copy())
            _df.to_csv(file)

            return _df

    def save_data(self, data: DataFrame, save_filename: str | Path):
        data.to_csv(save_filename)
