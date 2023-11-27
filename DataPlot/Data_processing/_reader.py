from pathlib import Path
from pandas import read_csv


def psd_reader(file_path=None):
    default_path = Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'distribution' / 'PNSD_dNdlogdp.csv'
    file_path = file_path or default_path

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')


def chemical_reader(file_path=None):
    default_path = Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'chemical.csv'
    file_path = file_path or default_path

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')[
               ['gRH', 'n_dry', 'n_amb', 'k_dry', 'k_amb', 'density', 'AS_volume_ratio', 'AN_volume_ratio',
                'OM_volume_ratio', 'Soil_volume_ratio', 'SS_volume_ratio', 'EC_volume_ratio', 'ALWC_volume_ratio']]


def sizedist_reader(file_path=None):
    default_path = Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'distribution'
    file_path = file_path or default_path

    with open(file_path / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(file_path / 'PSSD_dSdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PSSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(file_path / 'PVSD_dVdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PVSD = read_csv(f, parse_dates=['Time']).set_index('Time')

    return PNSD, PSSD, PVSD


def extdist_reader(file_path=None):
    default_path = Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'distribution'
    file_path = file_path or default_path

    with open(file_path / 'PESD_dextdlogdp_internal.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PESD_internal = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(file_path / 'PESD_dextdlogdp_external.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PESD_external = read_csv(f, parse_dates=['Time']).set_index('Time')

    return PESD_internal, PESD_external


def dry_extdist_reader(file_path=None):
    default_path = Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'distribution'
    file_path = file_path or default_path

    with open(file_path / 'PESD_dextdlogdp_dry_internal.csv', 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')