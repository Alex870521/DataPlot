import time
from functools import wraps


def timer(func=None):
    """ Decorator to measure the execution time of a function.

    This decorator calculates the elapsed time of the decorated function
    and prints the result in seconds.

    Parameters
    ----------
    - func (callable, optional): The function to be decorated.

    Returns
    -------
    - callable or None: If `func` is provided, returns the decorated function;
      otherwise, returns a decorator to be applied later.

    Examples
    --------
    1. Applying the decorator to a function directly:

    >>> @timer
    >>> def my_function():
    >>>     # Function logic here

    2. Using the decorator without providing a function:

    >>> @timer()
    >>> def my_function():
    >>>     # Function logic here

    Note
    ----
    The elapsed time is printed to the console after the function execution.

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