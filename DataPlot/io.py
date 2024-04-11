__all__ = ['load_default_data']


def load_default_data():
    import pandas as pd
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
        'OC':      [0.07, 0.95, 0.04, 0.05],
        'EC':      [0.20, 0.02, 0.85, 0.19],
        'Soil':    [0.20, 0.10, 0.07, 0.01],
        'SS':      [0.20, 0.10, 0.07, 0.01]
        }

    return pd.DataFrame(data, index=['Total', 'Clean', 'Transition', 'Event'])


if __name__ == '__main__':
    df = load_default_data()
