from pathlib import Path

from pandas import read_csv, concat

from DataPlot.process.script import (ImpactProc, ImproveProc, ChemicalProc, ParticleSizeDistProc,
                                     ExtinctionDistProc, OthersProc)


class DataBase:
    def __new__(cls, file_path, reset: bool = False, filename: Path | str = 'All_data.csv'):
        file_path = Path(file_path)

        print(f'\t\t \033[96m --- Loading Data --- \033[0m')
        if file_path.exists() and not reset:
            return read_csv(file_path, parse_dates=['Time'], na_values=('-', 'E', 'F'),
                            low_memory=False).set_index('Time')

        processor = [ImpactProc, ChemicalProc, ImproveProc, ParticleSizeDistProc, ExtinctionDistProc]
        reset = [False, False, False, False, False]
        save_filename = ['IMPACT.csv', 'chemical.csv', 'revised_IMPROVE.csv', 'PSD.csv', 'PESD.csv']

        _df = concat([processor().process_data(reset, save_filename) for processor, reset, save_filename in
                      zip(processor, reset, save_filename)], axis=1)

        # 6. others
        _df = OthersProc(reset=False, data=_df).process_data()

        # 7. save result
        _df.to_csv(file_path)

        return _df.copy()
