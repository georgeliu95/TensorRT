"""Common utils used by demo folder."""
import timeit
import logging

from os import listdir, rmdir
from shutil import rmtree
from typing import Callable, Union
from statistics import median


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


def measure_frameworks_inference_speed(
    stmt: Union[Callable, str], warmup: int = 3, number: int = 10, iterations: int = 10
) -> None:
    """
    Measures the time it takes to run frameworks code.
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
