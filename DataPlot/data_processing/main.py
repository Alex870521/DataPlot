from pathlib import Path
from pandas import read_csv, concat
from DataPlot.data_processing import *

PATH_MAIN = Path(__file__).parents[1] / 'Data-example'


@save_to_csv(PATH_MAIN / 'All_data.csv')
def main(reset=False, filename=None):
    # 1. EPB
    minion = DataReader('EPB.csv')

    # 2. IMPACT
    impact = ImpactProcessor(reset=False, filename='IMPACT.csv').process_data()

    # 3. Mass_volume
    chemical = ChemicalProcessor(reset=False, filename='chemical.csv').process_data()

    # 4. IMPROVE
    improve = ImproveProcessor(reset=False, filename='revised_IMPROVE.csv', version='revised').process_data()

    # 5. Number & Surface & volume & Extinction distribution
    PSD = SizeDist(reset=False, filename='PNSD_dNdlogdp.csv')

    psd = PSD.psd_process()
    ext = PSD.ext_process()

    # Extinction_PNSD_dry = dataproc.Extinction_dry_PSD_internal_process(reset=False)

    # 7. others
    _df = concat([minion, impact, chemical, improve, psd, ext], axis=1)
    df = other_process(_df.copy())
    return df


if __name__ == '__main__':
    df = main(reset=True)
