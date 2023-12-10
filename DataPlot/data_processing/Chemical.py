import numpy as np
import pandas as pd
from pandas import read_csv, concat
from DataProcessorBase import DataProcessorBase


# Note
# df['ALWC'] 不要加到 df_volume裡面


class ChemicalProcessor(DataProcessorBase):
    def __init__(self, reset=False, filename=None):
        super().__init__(reset)
        self.path = super().PATH_MAIN / 'Level2' / filename

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
        if pd.notna(_df['ALWC']):
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

        return _df['n_dry':]

    def process_data(self):
        if self.path.exists() and not self.reset:
            with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
                return read_csv(f, parse_dates=['Time']).set_index('Time')

        # 直接调用父类的属性
        df = concat([super().minion, super().impact], axis=1)

        df_mass = df[['NH4+', 'SO42-', 'NO3-', 'O_OC', 'Fe', 'Na+', 'O_EC', 'PM25']].dropna().copy().apply(self.mass,
                                                                                                           axis=1)
        df_volume = df_mass[['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'total_mass']].dropna().copy().apply(self.volume,
                                                                                                        axis=1)
        df_volume['ALWC'] = df['ALWC']
        df_vam = df_volume.copy().apply(self.volume_average_mixing, axis=1)

        return concat([df_mass, df_volume.drop(['ALWC'], axis=1), df_vam], axis=1).reindex(df.index.copy())

    def main(self):
        # Your main processing logic here
        _df = self.process_data()
        self.save_result(_df)
        return _df

    def save_result(self, data):
        # Your logic to save the result to a CSV file
        data.to_csv(self.path)


if __name__ == '__main__':
    result = ChemicalProcessor(reset=False, filename='chemical.csv').main()
