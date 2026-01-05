import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def timed(func):
    '''
    A decorator that logs the execution time of a function
    
    :param func: the function to time
    '''

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")

    return wrapper