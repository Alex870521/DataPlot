from pathlib import Path
from pandas import read_csv, concat
from DataPlot.data_processing import *

PATH_MAIN = Path(__file__).parents[1] / 'Data-example'


def main(reset=False, filename='All_data.csv'):

    file_path = PATH_MAIN / filename

    if file_path.exists() and not reset:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time'], low_memory=False).set_index('Time')
    else:
        # 1. EPB
        minion = DataReader('EPB.csv')

        # 2. IMPACT
        impact = ImpactProcessor(reset=False, filename='IMPACT.csv').process_data()

        # 3. Mass_volume
        chemical = ChemicalProcessor(reset=False, filename='chemical.csv').process_data()

        # 4. IMPROVE
        improve = ImproveProcessor(reset=False, filename='revised_IMPROVE.csv', version='revised').process_data()

        # 5. Number & Surface & volume & Extinction distribution
        PSD = SizeDist(reset=True, filename='PNSD_dNdlogdp.csv')

        psd = PSD.psd_process()
        ext = PSD.ext_process()

        # Extinction_PNSD_dry = dataproc.Extinction_dry_PSD_internal_process(reset=False)
        _df = concat([minion, impact, chemical, improve, psd, ext], axis=1)

        # 7. others
        _df = other_process(_df.copy())

        # 7. save result
        _df.to_csv(file_path)

        return _df


if __name__ == '__main__':
    df = main(reset=True)
