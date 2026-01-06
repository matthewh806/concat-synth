from typing import Callable, Iterable, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from .timer import timed
import logging

logger = logging.getLogger(__name__)

@timed
def run_parallel_cpu_tasks(
        task_function: Callable,
        tasks: Iterable,
        task_complete_callback: Optional[Callable] = None,
        max_workers = 4
):
    '''
    Parallelised task manager for running separate processes using a ProcessPoolExecutor.
    Note: the task_complete_callback should take the result object as a parameter
    
    :param task_function: The function to pass to the process executor
    :param tasks: Iterable of the inputs to be passed to the task function
    :param task_complete_callback: An optional method to call when a task is completed
    :param max_workers: The number of workers to parallelise this task with
    '''
    
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(task_function, task): task
            for task in tasks
        }

        num_tasks = len(tasks)
        for future in as_completed(futures):
            task = futures[future]
            try:
                completed += 1
                result = future.result()
                if task_complete_callback:
                    task_complete_callback(result)
                logger.info(f"Task ({completed} / {num_tasks}) completed for {task}")

            except Exception as e:
                logger.error(f"Task {task} generated an exception: {e}")
                continue


@timed
def run_parallel_io_tasks(
        task_function: Callable,
        tasks: Iterable,
        task_complete_callback: Optional[Callable] = None,
        max_workers = 4
):
    '''
    Parallelised task manager for running tasks in separate threads using a ThreadPoolExecutor.
    Note: the task_complete_callback should take the result object as a parameter
    
    :param task_function: The function to pass to the thread executor
    :param tasks: Iterable of the inputs to be passed to the task function
    :param task_complete_callback: An optional method to call when a task is completed
    :param max_workers: The number of workers to parallelise this task with
    '''
    
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(task_function, task): task
            for task in tasks
        }

        num_tasks = len(tasks)
        for future in as_completed(futures):
            task = futures[future]
            try:
                completed += 1
                result = future.result()
                if task_complete_callback:
                    task_complete_callback(result)
                logger.info(f"Task ({completed} / {num_tasks}) completed for {task}")

            except Exception as e:
                logger.error(f"Task {task} generated an exception: {e}")
                continue