from pathlib import Path
from pandas import read_csv, concat
from DataPlot.data_processing import *

PATH_MAIN = Path(__file__).parents[2] / 'Data-example'


@save_to_csv(PATH_MAIN / 'All_data.csv')
def main(reset=False, filename=None):
    # 1. EPB
    minion = DataReader('EPB.csv')

    # 2. IMPACT
    impact = ImpactProcessor(reset=True, filename='IMPACT.csv').process_data()

    # 3. Mass_volume
    chemical = ChemicalProcessor(reset=True, filename='chemical.csv').process_data()

    # 4. IMPROVE
    improve = ImproveProcessor(reset=True, filename='revised_IMPROVE.csv', version='revised').process_data()

    # 5. Number & Surface & volume distribution
    PSD = SizeDist(reset=True, filename='PNSD_dNdlogdp.csv').psd_process()

    # 6. Extinction distribution
    PESD = SizeDist(reset=True, filename='PNSD_dNdlogdp.csv').ext_process()
    # Extinction_PNSD_dry = dataproc.Extinction_dry_PSD_internal_process(reset=False)

    return concat([minion, impact, chemical, improve, PESD, PSD], axis=1)


if __name__ == '__main__':
    df = main(reset=True)
