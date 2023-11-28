import time
from functools import wraps


def timer(func=None):
    """ 輸出函式耗時

    :param func:
    :return:
    """
    def decorator(_func):
        @wraps(_func)
        def wrapper(*args, **kwargs):
            st = time.perf_counter()
            result = _func(*args, **kwargs)
            print('time cost: {:.2f} seconds'.format(time.perf_counter() - st))
            return result

        return wrapper

    if func is None:
        return decorator

    else:
        return decorator(func)