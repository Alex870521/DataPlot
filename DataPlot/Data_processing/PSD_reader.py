from pathlib import Path
from pandas import read_csv


def psd_reader(file_path=None):
    default_path = Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'distribution' / 'PNSD_dNdlogdp.csv'

    file_path = file_path or default_path

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')


def chemical_reader():
    with open(Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'chemical.csv', 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')[
               ['gRH', 'n_dry', 'n_amb', 'k_dry', 'k_amb', 'density', 'AS_volume_ratio', 'AN_volume_ratio',
                'OM_volume_ratio', 'Soil_volume_ratio', 'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']]