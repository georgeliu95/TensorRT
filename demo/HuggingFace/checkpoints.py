"""
Helper file for generating common checkpoints.
"""

# TRT-HuggingFace
from typing import List
from networks import NetworkMetadata

# externals
import toml


class NNTomlCheckpoint:
    """Loads a toml checkpoint file for comparing labels and inputs."""

    def __init__(self, fpath: str, network_name: str, metadata: NetworkMetadata):
        """Loads the toml file for processing."""
        data = {}
        with open(fpath) as f:
            data = toml.load(f)

        # Select the current input data
        # try to get the base data
        network = data.get(network_name, {})
        self.data = network.get("default", {})
        # Defaults are also used as baselines for the network in case there are deviations known in variants.
        self.baseline = self.data.copy()

        # then apply specific data
        addendum = network.get(metadata.variant, {})
        self.data = {
            k: {**self.data[k], **addendum.get(k, {})} for k in self.data.keys()
        }

    def _iterate_data(self, slice: List[str], skip_keyword: str = "skip"):
        """
        Helper for child classes to iterate through a slice of data.

        Return:
            (Union[Dict[str, str], List[str]]): Returns a list of all value keys given in 'slice' or if more than one value is given for 'slice' then a dictionary instead.
        """
        returns_dict = len(slice) > 1
        for value in self.data.values():
            if "skip" in value:
                continue

            if returns_dict:
                yield {s: value[s] for s in slice}
            else:
                yield value[slice[0]]


class NNSemanticCheckpoint(NNTomlCheckpoint):
    """Requires the following data structure:

    [<network>.<variant>]
        [input_a]
        label = "sample_label"
        input = "sample_input"

        [input_b]
        label = "sample_label"
        input = "sample_input"
    """

    def __iter__(self):
        return self._iterate_data(["label", "input"])

    def labels(self):
        return self._iterate_data(["label"])

    def inputs(self):
        return self._iterate_data(["input"])
