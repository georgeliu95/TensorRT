import subprocess as sp
import sys
import os

from polygraphy.logger import G_LOGGER

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
BIN_DIR = os.path.join(ROOT_DIR, "bin")
polygraphy = os.path.join(BIN_DIR, "polygraphy")


def check_subprocess(status):
    assert not status.returncode


def run_subtool(subtool, additional_opts):
    cmd = [sys.executable, polygraphy, subtool] + additional_opts + ["-vvvvv"]
    G_LOGGER.info("Running command: {:}".format(" ".join(cmd)))
    check_subprocess(sp.run(cmd))


def run_polygraphy_run(additional_opts=[]):
    run_subtool("run", additional_opts)


def run_polygraphy_inspect(additional_opts=[]):
    run_subtool("inspect", additional_opts)


def run_polygraphy_precision(additional_opts=[]):
    run_subtool("precision", additional_opts)


def run_polygraphy_surgeon(additional_opts=[]):
    run_subtool("surgeon", additional_opts)
