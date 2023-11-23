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
            df = read_csv(f, parse_dates=['Time'], low_memory=False).set_index('Time')
        return df

    if isinstance(filename, tuple) and all([file.exists() for file in filename]):
        raise TypeError("Please check the files or make sure you're input a DataFrame or a tuple of DataFrames.")

    else:
        raise TypeError("Please check the files or make sure you're input a DataFrame or a tuple of DataFrames.")


def open_csvs(filename):
    """開啟一個或多個csv並回傳一個或多個Dataframe

    :param filename: Path
    :return: Dataframe, tuple(Dataframe)
    """
    if isinstance(filename, Path) and filename.exists():
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            df = read_csv(f, parse_dates=['Time'], low_memory=False).set_index('Time')
        return df

    if isinstance(filename, tuple) and all([file.exists() for file in filename]):
        dfs = ()
        for file in filename:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                df = read_csv(f, parse_dates=['Time']).set_index('Time')
            dfs += (df,)
        return dfs

    else:
        raise FileNotFoundError("Please check the files or make sure you're input a DataFrame or a tuple of DataFrames.")


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
                print('Finish....', _func.__name__)
                return open_csv(filename)

            result = _func(reset=True, filename=filename, *args, **kwargs)

            if isinstance(result, pd.DataFrame):
                result.to_csv(filename)
                path, name = os.path.split(filename)
                print('Export....', name, 'to', path)

            elif isinstance(result, tuple):
                for i, (df, file) in enumerate(zip(result, filename)):
                    df.to_csv(file)
                    path, name = os.path.split(file)
                    print('Export....', name, 'to', path)

            else:
                raise TypeError("The function must return a DataFrame or a tuple of DataFrames")

            print('Finish....', _func.__name__)
            return result
        return wrapper
    return decorator
