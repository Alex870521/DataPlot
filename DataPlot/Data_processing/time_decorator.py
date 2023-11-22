import time
from functools import wraps


def timer(func=None, *, print_args=False):
    """ 輸出函式耗時

    :param func:
    :param print_args:
    :return:
    """
    def decorator(_func):
        @wraps(_func)
        def wrapper(*args, **kwargs):
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