"""
Helper files containing several namedtuples that communicate between individual scripts.
"""

# std
import os
import shutil
import string

from typing import List
from collections import namedtuple

# helpers
from utils import remove_if_empty

FILENAME_VALID_CHARS = "-~_.() {}{}".format(string.ascii_letters, string.digits)

"""NNDTResult(polygraphy: NetworkResult, trtexec: NetworkResult, frameworks: NetworkResult)"""
# TODO: Used by testing framework, do not remove yet
NNDTResult = namedtuple("NNDTResult", ["polygraphy", "trtexec", "frameworks"])

"""NetworkResult(output_tensor: np.array, semantic_output: np.array, median_runtime: float, model_location: [str])"""
NetworkResult = namedtuple(
    "NetworkResult", ["output_tensor", "semantic_output", "median_runtime", "models"]
)

# Tracks TRT Precision Config
"""Precision(fp16: Bool, int8: Bool, tf32: Bool)"""
Precision = namedtuple("Precision", ["fp16", "int8"])

"""NetworkMetadata(variant: str, precision: Precision, other: Union[namedtuple, None])"""
NetworkMetadata = namedtuple("NetworkMetadata", ["variant", "precision", "other"])

"""TimingProfile(iterations: int, repeat: int)"""
TimingProfile = namedtuple("TimingProfile", ["iterations", "number", "warmup"])

"""
String encodings to genereted network models.
    NetworkModels(torch: Tuple[str], onnx: Tuple[str])
"""
NetworkModels = namedtuple("NetworkModels", ["torch", "onnx"])


class NNFolderWorkspace:
    """For keeping track of workspace folder and for cleaning them up."""

    def __init__(self, network_name, metadata, working_directory):
        self.rootdir = working_directory
        self.metadata = metadata
        self.network_name = network_name
        self.dpath = os.path.join(self.rootdir, self.network_name)
        os.makedirs(self.dpath, exist_ok=True)

    def get_path(self):
        dpath = os.path.join(self.rootdir, self.network_name)
        return dpath

    def cleanup(self, force_remove=False):
        fpath = self.get_path()
        if force_remove:
            return shutil.rmtree(fpath)
        remove_if_empty(
            fpath,
            success_msg="Sucessfully removed workspace.",
            error_msg="Unable to remove workspace.",
        )


# Config Class
class NNConfig:
    """Contains info for a given network that we support."""

    def __init__(self, network_name, variants=None):
        assert self._is_valid_filename(
            network_name
        ), "Network name: {} is not filename friendly.".format(network_name)

        self.network_name = network_name
        self.variants = variants

        # Due to limitations of namedtuples and pickle function, namedtupled must be tracked as an instance
        # which refers to a global.
        if len(self.variants) > 0:
            self.MetadataClass = type(self.variants[0].other)
        else:
            self.MetadataClass = None

    def _is_valid_filename(self, filename: str) -> bool:
        """
        Checks if a given filename is valid, helpful for cross platform dependencies.
        """
        return all(c in FILENAME_VALID_CHARS for c in filename)

    def get_python_requirements():
        return []

    def get_metadata_string(self, metadata: NetworkMetadata) -> str:
        """
        Serializes a Metadata object into string.
        String will be checked if friendly to filenames across Windows and Linux operating systems.

        returns:
            string: <network>-<variant-name>-<precision>-<others>
        """

        precision_str = "-".join(
            [k for k, v in metadata.precision._asdict().items() if v]
        )
        result = [self.network_name, metadata.variant]
        if precision_str:
            result.append(precision_str)

        other_result = [
            "{}~{}".format(k, str(v)) for k, v in metadata.other._asdict().items()
        ]
        # Remove all boolean values that are False
        other_result_filtered = [v for v in other_result if "~False" not in v]
        # We use the += operator as other_result_filtered may be empty. Saves an if statement
        result += "-".join(other_result_filtered)

        final_str = "-".join(result)
        assert self._is_valid_filename(
            final_str
        ), "Metadata for current network {} is not filename friendly: {}.".format(
            self.network_name, final_str
        )

        return final_str
