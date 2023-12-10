from pathlib import Path
from pandas import read_csv, concat
from DataPlot.data_processing.decorator import save_to_csv, timer

PATH_MAIN = Path(__file__).parents[2] / 'Data-example'


@save_to_csv(PATH_MAIN / 'Level1' / 'IMPACT.csv')
def impact_process(filename=None, reset=False):
    if filename.exists() & (~reset):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'Optical.csv', 'r', encoding='utf-8', errors='ignore') as f:
        optical = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'PBLH.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PBLH = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'OCEC.csv', 'r', encoding='utf-8', errors='ignore') as f:
        OCEC = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'Teom.csv', 'r', encoding='utf-8', errors='ignore') as f:
        Teom = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'ISORROPIA.csv', 'r', encoding='utf-8', errors='ignore') as f:
        ISOR = read_csv(f, parse_dates=['Time']).set_index('Time')

    return concat([optical, PBLH, OCEC, Teom, ISOR], axis=1)


if __name__ == '__main__':
    df = impact_process(reset=False)

