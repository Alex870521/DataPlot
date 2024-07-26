from typing import Literal

from pandas import DataFrame, read_csv

from DataPlot.dataProcess import *
from DataPlot.plot import Color, Unit, set_figure
from DataPlot.process import DataBase, DataReader, DataClassifier
from DataPlot.rawDataReader import RawDataReader


def load_default_chemical_data():
    # The following data is from the chemical composition of real atmospheric particles.
    #
    # The six main chemical components that comprised PM2.5 are listed in the data.
    # Here, we test the radar charts to see if we can clearly identify how the
    # chemical components vary between the three pollutant scenarios:
    #
    #  1) Whole sampling period (Total)
    #  2) Clean period (Clean)
    #  3) Transition period (Transition)
    #  4) Event period (Event)

    data = {
        'Sulfate': [0.01, 0.34, 0.02, 0.71],
        'Nitrate': [0.88, 0.13, 0.34, 0.13],
        'OC': [0.07, 0.95, 0.04, 0.05],
        'EC': [0.20, 0.02, 0.85, 0.19],
        'Soil': [0.20, 0.10, 0.07, 0.01],
        'SS': [0.20, 0.10, 0.07, 0.01]
    }

    return DataFrame(data, index=['Total', 'Clean', 'Transition', 'Event'])


def load_dataset(dataset_name: Literal["Tunghai", "Taipei", "Iris"] = "Tunghai") -> DataFrame:
    dataset_uris = {
        "Tunghai": "https://github.com/Alex870521/DataPlot/blob/main/DataPlot/data/All_data.csv",
        "Taipei": "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
        "Iris": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    }

    # Ensure the dataset name is valid
    if dataset_name not in dataset_uris:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    uri = dataset_uris[dataset_name]

    # Load and return the dataset
    return read_csv(uri, header=None if dataset_name == "Iris" else "infer")
