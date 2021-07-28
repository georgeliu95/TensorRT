"""Common utils used by demo folder."""

# std
import timeit
import logging
import inspect

from os import listdir, rmdir
from shutil import rmtree
from typing import Callable, Union
from statistics import median

# pytorch
import torch

# IO #
def confirm_folder_delete(
    fpath: str, prompt: str = "Confirm you want to delete entire folder?"
) -> None:
    """
    Confirms whether or not user wants to delete given folder path.

    Args:
        fpath (str): Path to folder.
        prompt (str): Prompt to display

    Returns:
        None
    """
    msg = prompt + " {} [Y/n] ".format(fpath)
    confirm = input(msg)
    if confirm == "Y":
        rmtree(fpath)
    else:
        logging.info("Skipping file removal.")


def remove_if_empty(
    fpath: str,
    success_msg: str = "Folder successfully removed.",
    error_msg: str = "Folder cannot be removed, there are files.",
) -> None:
    """
    Removes an entire folder if folder is empty. Provides print info statements.

    Args:
        fpath: Location to folder
        success_msg: Success message.
        error_msg: Error message.

    Returns:
        None
    """
    if len(listdir(fpath)) == 0:
        rmdir(fpath)
        logging.info(success_msg + " {}".format(fpath))
    else:
        logging.info(error_msg + " {}".format(fpath))


def measure_python_inference_code(
    stmt: Union[Callable, str], warmup: int = 3, number: int = 10, iterations: int = 10
) -> None:
    """
    Measures the time it takes to run Pythonic inference code.
    Statement given should be the actual model inference like forward() in torch.

    See timeit for more details on how stmt works.

    Args:
        stmt (Union[Callable, str]): Callable or string for generating numbers.
        number (int): Number of times to call function per iteration.
        iterations (int): Number of measurement cycles.
    """
    logging.debug(
        "Measuring inference call with warmup: {} and number: {} and iterations {}".format(
            warmup, number, iterations
        )
    )
    # Warmup
    warmup_mintime = timeit.repeat(stmt, number=number, repeat=warmup)
    logging.debug("Warmup times: {}".format(warmup_mintime))

    return median(timeit.repeat(stmt, number=number, repeat=iterations))

def send_to_device(*args):
    """Sends all given pytorch variables to cuda device."""
    results = []
    for a in args:
        results.append(a.to("cuda"))

    if len(args) == 1:
        return results[0]
    else:
        return results

# Function Decorators #
def use_cuda(func: Callable):
    """
    Tries to send all parameters of a given function to cuda device if user supports it.
    Object must have a "to(device: str)" and maps to target device "cuda"
    Basically, uses torch implementation.

    Wrapped functions musts have keyword argument "use_cuda: bool" which enables
    or disables toggling of cuda.
    """

    def wrapper(*args, **kwargs):
        caller_kwargs = inspect.getcallargs(func, *args, **kwargs)
        assert "use_cuda" in caller_kwargs, "Function must have 'use_cuda' as a parameter."

        if caller_kwargs["use_cuda"] and torch.cuda.is_available():
            new_kwargs = {}
            for k, v in caller_kwargs.items():
                if getattr(v, "to", False):
                    new_kwargs[k] = v.to("cuda")
                else:
                    new_kwargs[k] = v

            return func(**new_kwargs)
        else:
            return func(**caller_kwargs)

    return wrapper
