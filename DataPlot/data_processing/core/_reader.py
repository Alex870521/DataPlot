from pathlib import Path
from pandas import read_csv, read_json, read_table


class DataReader:
    def __init__(self, file_path=None, default_path=None):
        self.file_path = file_path or default_path

    def read_csv(self):
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time']).set_index('Time')

    def read_json(self):
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return read_json(f)

    def read_table(self, **kwargs):
        return read_table(self.file_path, **kwargs)


class PSDReader(DataReader):
    def __init__(self, file_path=None):
        default_path = Path(__file__).parents[2] / 'Data-example' / 'Level2' / 'distribution' / 'PNSD_dNdlogdp.csv'
        super().__init__(file_path, default_path)


def psd_reader(file_path=None):
    default_path = Path(__file__).parents[2] / 'Data-example' / 'Level2' / 'distribution' / 'PNSD_dNdlogdp.csv'
    file_path = file_path or default_path

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')


def chemical_reader(file_path=None):
    default_path = Path(__file__).parents[2] / 'Data-example' / 'Level2' / 'chemical.csv'
    file_path = file_path or default_path

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')[
               ['gRH', 'n_dry', 'n_amb', 'k_dry', 'k_amb', 'density', 'AS_volume_ratio', 'AN_volume_ratio',
                'OM_volume_ratio', 'Soil_volume_ratio', 'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']]


def sizedist_reader(file_path=None):
    default_path = Path(__file__).parents[2] / 'Data-example' / 'Level2' / 'distribution'
    file_path = file_path or default_path

    with open(file_path / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(file_path / 'PSSD_dSdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PSSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(file_path / 'PVSD_dVdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PVSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    return PNSD, PSSD, PVSD


def extdist_reader(file_path=None):
    default_path = Path(__file__).parents[2] / 'Data-example' / 'Level2' / 'distribution'
    file_path = file_path or default_path

    with open(file_path / 'PESD_dextdlogdp_internal.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PESD_internal = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(file_path / 'PESD_dextdlogdp_external.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PESD_external = read_csv(f, parse_dates=['Time']).set_index('Time')

    return PESD_internal, PESD_external


def dry_extdist_reader(file_path=None):
    default_path = Path(__file__).parents[2] / 'Data-example' / 'Level2' / 'distribution'
    file_path = file_path or default_path

    with open(file_path / 'PESD_dextdlogdp_dry_internal.csv', 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')