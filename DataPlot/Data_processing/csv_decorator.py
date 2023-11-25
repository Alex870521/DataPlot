import os
import pandas as pd
from pathlib import Path
from functools import wraps
from pandas import read_csv


def open_csv(filename):
    """開啟csv並回傳Dataframe

    :param filename: Path
    :return: Dataframe
    """
    if isinstance(filename, Path) and filename.exists():
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time'], low_memory=False).set_index('Time')

    elif not filename.exists():
        raise FileNotFoundError(f"The {filename} was not found. Please change the reset to 'True'")

    else:
        raise TypeError("Please check the files or make sure you're input a DataFrame or a tuple of DataFrames.")


def save_to_csv(filename):
    """ 當Datafrmae作為一個方法的輸出變數即可使用這個方法儲存csv

    :param filename:
    :return:
    """
    def decorator(_func):
        @wraps(_func)
        def wrapper(reset=False, *args, **kwargs):
            print('Loading...', _func.__name__)
            if not reset:
                print('Opening...', filename)
                return open_csv(filename)

            result = _func(reset=True, filename=filename, *args, **kwargs)

            if isinstance(result, pd.DataFrame):
                result.to_csv(filename)
                path, name = os.path.split(filename)
                print('Export....', name, 'to', path)

            else:
                raise TypeError("The function must return a DataFrame or a tuple of DataFrames")

            print('Finish....', _func.__name__)
            return result
        return wrapper
    return decorator
