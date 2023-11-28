from pathlib import Path
from pandas import read_csv, concat
from DataPlot.Data_processing.decorator.csv_decorator import save_to_csv
import DataPlot.Data_processing as dataproc

PATH_MAIN = Path(__file__).parent.parent.parent / 'Data'


@save_to_csv(PATH_MAIN / 'All_data.csv')
def main(reset=False, filename=None):
    # 1. EPB
    with open(PATH_MAIN / 'level1' / 'EPB.csv', 'r', encoding='utf-8', errors='ignore') as f:
        minion = read_csv(f, parse_dates=['Time'], na_values=['-', 'E', 'F']).set_index('Time')

    # 2. IMPACT
    impact = dataproc.impact_process(reset=False)

    # 3. Mass_volume
    chemical = dataproc.chemical_process(reset=False)

    # 4. IMPROVE
    improve = dataproc.improve_process(reset=False, version='revised')

    # 5. Number & Surface & volume distribution
    PSD = dataproc.SizeDist().psd_process()

    # 6. Extinction distribution
    PESD = dataproc.SizeDist().ext_process()
    # Extinction_PNSD_dry = dataproc.Extinction_dry_PSD_internal_process(reset=False)

    # df = dataproc.other_process(df.copy())

    return concat([minion, impact, chemical, improve, PESD, PSD], axis=1)


if __name__ == '__main__':
    df = main(reset=True)
