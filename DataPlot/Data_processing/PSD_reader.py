from pathlib import Path
from pandas import read_csv


def _reader():
    PATH_MAIN = Path(__file__).parent.parent.parent / 'Data' / 'Level2'
    PATH_DIST = PATH_MAIN / 'distribution'
    with open(PATH_DIST / 'PNSD_dNdlogdp.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PNSD = read_csv(f, parse_dates=['Time']).set_index('Time')
    return PNSD


if __name__ == '__main__':
    _reader()