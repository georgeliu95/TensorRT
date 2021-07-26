"""Interface classes required for each registered network script."""

import argparse

from abc import ABCMeta, abstractmethod
import logging
from networks import NetworkResult, NetworkMetadata, NNConfig, TimingProfile
from typing import List


class NetworkCommand(metaclass=ABCMeta):
    """Base class that each network script's command module should inherit."""

    description = "NetworkCommand"

    def __init__(self, network_config: NNConfig, description: str):
        self.config = network_config()
        self.inference_input = []
        self.description = description
        self._parser = argparse.ArgumentParser(description=description)

    def __call__(self):
        self.add_args(self._parser)
        self.args = self._parser.parse_args()

        if self.args.verbose:
            logging.basicConfig(level=logging.DEBUG)

        self.metadata = self.args_to_network_metadata(self.args)
        self.check_network_metadata_is_supported(self.metadata)

    def add_args(self, parser) -> argparse.ArgumentParser:
        general_group = parser.add_argument_group("general")
        general_group.add_argument(
            "--verbose", help="Display verbose logs.", action="store_true"
        )

    def check_network_metadata_is_supported(self, metadata: NetworkMetadata) -> None:
        """
        Checks if current command supports the given metadata as defined by the NNConfig.
        Args:
            metadata (NetworkMetadata): NetworkMetadata to check if input is supported.

        Throws:
            NotImplementedError: If the given metadata is not a valid configuration for this network.

        Returns:
            None
        """
        if metadata not in self.config.variants:
            raise NotImplementedError("The following network config is not yet supported by our scripts: {}".format(metadata))

    @abstractmethod
    def args_to_network_metadata(self, args) -> NetworkMetadata:
        pass


class FrameworkCommand(NetworkCommand):
    """Base class that is associated with Frameworks related scripts."""

    @abstractmethod
    def run_framework(
        self,
        metadata: NetworkMetadata,
        network_input: List[str],
        working_directory: str,
        save_onnx_model: bool,
        save_pytorch_model: bool,
        timing_profile: TimingProfile,
    ) -> List[NetworkResult]:
        pass

    def __call__(self):
        super().__call__()
        return self.run_framework(
            self.metadata,
            self.inference_input,
            self.args.working_dir,
            self.args.save_onnx_model,
            self.args.save_torch_model,
            TimingProfile(
                iterations=self.args.iterations,
                number=self.args.number,
                warmup=self.args.warmup,
            ),
        )

    def add_args(self, parser) -> argparse.ArgumentParser:
        super().add_args(parser)
        timing_group = parser.add_argument_group("inference measurement")
        timing_group.add_argument(
            "--iterations", help="Number of iterations to measure.", default=10
        )
        timing_group.add_argument(
            "--number",
            help="Number of actual inference cycles per iterations.",
            default=10,
        )
        timing_group.add_argument(
            "--warmup",
            help="Number of warmup iterations before actual measurement occurs.",
            default=3,
        )


class PolygraphyCommand(NetworkCommand):
    """Base class that is associated with Polygraphy related scripts."""

    @abstractmethod
    def run_polygraphy() -> List[NetworkResult]:
        # TODO: Still need to define
        pass

    def __call__(self):
        super().__call__()


class TRTCommand(NetworkCommand):
    """Base class that is associated with TensorRT related scripts."""

    @abstractmethod
    def run_tensorrt() -> List[NetworkResult]:
        # TODO: Still need to define interface
        pass

    def __call__(self):
        super().__call__()
