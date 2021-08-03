"""
Demonstrates TensorRT capabilities with networks located in HuggingFace repository.
Requires Python 3.5+
"""

import os
import sys
import glob
import logging
import argparse
import importlib

from abc import abstractmethod
from typing import List

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Wrapper actions supported
WRAPPER_RUN_ACTION = "run"
WRAPPER_LIST_ACTION = "list"
WRAPPER_COMPARE_ACTION = "compare"
WRAPPER_ACTIONS = [
    WRAPPER_RUN_ACTION,
    WRAPPER_LIST_ACTION,
    WRAPPER_COMPARE_ACTION
]

class Action:
    def __init__(self, networks, parser):
        self.networks = networks
        self.parser = parser
        self.add_args(self.parser)

    @abstractmethod
    def execute(self, args):
        pass

    @abstractmethod
    def add_args(self, parser):
        pass

class NetworkScriptAction(Action):

    # Reserved files names for each network folder
    FRAMEWORKS_SCRIPT_NAME = "frameworks"
    POLYGRAPHY_SCRIPT_NAME = "trt_polygraphy"
    PER_NETWORK_SCRIPTS = [
        FRAMEWORKS_SCRIPT_NAME,
        POLYGRAPHY_SCRIPT_NAME
    ]

    def add_args(self, parser):
        network_group = parser.add_argument_group("specify network")
        network_group.add_argument("network", help="Network to run.", choices=self.networks)

    def load_script(self, script_name: str, args: argparse.Namespace):
        """Helper for loading a specific script for given network."""
        assert script_name in self.PER_NETWORK_SCRIPTS, "Script must be a reserved name."

        # Load the specific commandline script
        return importlib.import_module("{}.{}".format(args.network, script_name))


class RunAction(NetworkScriptAction):
    def execute(self, args):
        module = self.load_script(args.script, args)
        module.RUN_CMD._parser = self.parser
        os.chdir(args.network)
        print(module.RUN_CMD())

    def add_args(self, parser):
        super().add_args(parser)
        run_group = parser.add_argument_group("run args")
        run_group.add_argument("script", choices=self.PER_NETWORK_SCRIPTS)

class CompareAction(NetworkScriptAction):
    pass

class ListAction(Action):
    def __init__(self, networks):
        self.networks = networks

    def execute(self, args):
        print("Networks that are supported by HuggingFace Demo:")
        [print(n) for n in self.networks]
        return 0

def get_action(
    action_name: str,
    networks: List[str],
    parser: argparse.ArgumentParser
) -> Action:
    return {
        WRAPPER_COMPARE_ACTION: CompareAction,
        WRAPPER_LIST_ACTION: ListAction,
        WRAPPER_RUN_ACTION: RunAction
    }[action_name](networks, parser)


def get_default_parser(
    networks: List[str], description: str = "", add_default_help=False
) -> argparse.ArgumentParser:
    """
    Returns argparser for use by main(). Allows the ability to toggle default help message with a custom help flag
    so that argparser does not throw SystemExit when --help is passed in. Useful for custom --help functionality.

    Returns:
        (argparse.ArgumentParser): argparser used by main()
    """
    # This variable is set so that usage errors don't show up in wrapper
    parser = argparse.ArgumentParser(description=description, add_help=add_default_help)
    required_group = parser.add_argument_group("required wrapper arguments")

    required_group.add_argument(
        "action", choices=WRAPPER_ACTIONS
    )

    if not add_default_help:
        parser.add_argument(
            "--help",
            "-h",
            help="Shows help message. If --network is supplied, returns help for specific script.",
            action="store_true",
        )
    return parser


def main() -> None:
    """
    Parses network folders and responsible for passing --help flags to subcommands if --network is provided.

    Returns:
        None
    """
    # Get all available network scripts
    networks = []
    for network_configs in glob.glob(os.path.join(ROOT_DIR, "*", "*Config.py")):
        network_name = os.path.split(os.path.split(network_configs)[0])[1]
        networks.append(network_name)

    # Add network folder for entry point
    description = (
        "Runs TensorRT networks that are based-off of HuggingFace variants."
    )
    parser = get_default_parser(networks, description, add_default_help=False)

    # Get the general network wrapper help
    known_args, _ = parser.parse_known_args()

    # Delegate parser to action specifics
    action = get_action(known_args.action, networks, parser)
    known_args, _ = parser.parse_known_args()
    return action.execute(known_args)

if __name__ == "__main__":
    main()
