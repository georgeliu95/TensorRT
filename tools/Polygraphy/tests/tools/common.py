import subprocess as sp
import sys
import os

from polygraphy.logger import G_LOGGER

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
BIN_DIR = os.path.join(ROOT_DIR, "bin")
polygraphy = os.path.join(BIN_DIR, "polygraphy")


def check_subprocess(status):
    assert not status.returncode


def run_subtool(subtool, additional_opts, disable_verbose=False):
    cmd = [sys.executable, polygraphy, subtool] + additional_opts
    if not disable_verbose:
        cmd += ["-vvvvv"]
    G_LOGGER.info("Running command: {:}".format(" ".join(cmd)))
    status = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    check_subprocess(status)
    return status


def run_polygraphy_run(additional_opts=[], disable_verbose=False):
    return run_subtool("run", additional_opts, disable_verbose)


def run_polygraphy_inspect(additional_opts=[], disable_verbose=False):
    return run_subtool("inspect", additional_opts, disable_verbose)


def run_polygraphy_precision(additional_opts=[], disable_verbose=False):
    return run_subtool("precision", additional_opts, disable_verbose)


def run_polygraphy_surgeon(additional_opts=[], disable_verbose=False):
    return run_subtool("surgeon", additional_opts, disable_verbose)
