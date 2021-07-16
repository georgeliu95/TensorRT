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

from typing import List

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


def get_parser(
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

    # We purposely don't toggle "required" parameter here because argparse will force an exit with usage which
    # won't allow us to parse "--help" using custom logic.
    required_group.add_argument(
        "--network", "-n", help="Network name.", choices=networks
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
        "Runs TensorRT networks that are based-off of HuggingFace network variants."
    )
    parser = get_parser(networks, description, add_default_help=False)

    # Get the general network wrapper help
    known_args, _ = parser.parse_known_args()

    # Get network value
    network_selected = None
    network_selected = known_args.network
    if network_selected is None:
        parser.print_help()
        return

    # Load the specific commandline script
    framework_module = importlib.import_module("{}.frameworks".format(network_selected))

    # Override parser to get an prepended help message
    description = description + " {}".format(framework_module.RUN_CMD.description)
    framework_module.RUN_CMD._parser = get_parser(
        networks, description=description, add_default_help=True
    )
    framework_module.RUN_CMD()


if __name__ == "__main__":
    main()
