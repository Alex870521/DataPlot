from pathlib import Path
from pandas import read_csv


def _reader():
    with open(Path(__file__).parent.parent.parent / 'Data' / 'Level2' / 'distribution' / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        return read_csv(f, parse_dates=['Time']).set_index('Time')
