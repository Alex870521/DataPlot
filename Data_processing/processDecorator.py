import os
import time
from pathlib import Path
import pandas as pd
from functools import wraps
from pandas import read_csv


def function_handler(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        print('Loading...', func.__name__)
        results = func(*args, **kwargs)
        print('Finish...', func.__name__)
        return results
    return wrap


def timer(func=None, *, print_args=False):
    """ 輸出函式耗時

    :param func:
    :param print_args:
    :return:
    """
    def decorator(_func):
        @wraps(_func)
        def wrapper(reset=False, *args, **kwargs):
            st = time.perf_counter()
            result = _func(*args, **kwargs)
            if print_args:
                print(f'"{_func.__name__}, args: {args}, kwargs: {kwargs}"')
            print('time cost: {} seconds'.format(time.perf_counter() - st))
            return result

        return wrapper

    if func is None:
        return decorator

    else:
        return decorator(func)


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
    def decorator(func):
        @wraps(func)
        def wrapper(reset=False, *args, **kwargs):
            print('Loading...', func.__name__)
            if not reset:
                print('Finish....', func.__name__)
                return open_csv(filename)

            result = func(reset=True, filename=filename, *args, **kwargs)

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

            print('Finish....', func.__name__)
            return result
        return wrapper
    return decorator


if __name__ == '__main__':
    PATH_MAIN = Path("/Data_processing")

    # @save_to_csv((PATH_MAIN / 'output1.csv', PATH_MAIN / 'output2.csv'))
    @save_to_csv(PATH_MAIN / 'output1.csv')
    def my_function(filename=None, reset=False):
        # 這個函式返回一個或多個DataFrame
        df1 = pd.DataFrame({'Time': [pd.Timestamp('2020-04-11 01:00:00'), pd.Timestamp('2020-04-11 02:00:00'), pd.Timestamp('2020-04-11 03:00:00')],
                            'A': [4, 5, 6], 'B': [4, 5, 6]}).set_index('Time')
        # df2 = pd.DataFrame({'Time': [pd.Timestamp('2020-04-11 01:00:00'), pd.Timestamp('2020-04-11 02:00:00'), pd.Timestamp('2020-04-11 03:00:00')],
        # 'C': [4, 5, 6], 'D': [10, 11, 12]}).set_index('Time')
        return df1


    abc = my_function(reset=True)
