from pathlib import Path
from pandas import read_csv, concat
from Data_processing.processDecorator import save_to_csv

PATH_MAIN = Path("C:/Users/alex/PycharmProjects/DataPlot/Data")


@save_to_csv(PATH_MAIN / 'Level1' / 'IMPACT.csv')
def impact_process(reset=False, filename=None):
    if filename.exists() & (~reset):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'optical.csv', 'r', encoding='utf-8', errors='ignore') as f:
        optical = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'PBLH.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PBLH = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'OCEC.csv', 'r', encoding='utf-8', errors='ignore') as f:
        OCEC = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'PM1.csv', 'r', encoding='utf-8', errors='ignore') as f:
        PM1 = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'Level1' / 'ISORROPIA.csv', 'r', encoding='utf-8', errors='ignore') as f:
        ISOR = read_csv(f, parse_dates=['Time']).set_index('Time')

    df = concat([optical, PBLH, OCEC, PM1, ISOR], axis=1)

    return df


if __name__ == '__main__':
    df = impact_process(reset=False)

