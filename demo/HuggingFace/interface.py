"""Interface classes required for each registered network script."""

import argparse

import logging
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

# polygraphy
from polygraphy.logger import G_LOGGER

from networks import (
    NetworkResult,
    NetworkMetadata,
    NNConfig,
    NetworkModel,
    TimingProfile,
)
from checkpoints import NNSemanticCheckpoint


class NetworkCommand(metaclass=ABCMeta):
    """Base class that each network script's command module should inherit."""

    description = "NetworkCommand"

    def __init__(self, network_config: NNConfig, description: str):
        self.config = network_config()
        self.description = description
        self._parser = argparse.ArgumentParser(description=description)

    def __call__(self):
        self.add_args(self._parser)
        self._args = self._parser.parse_args()

        if self._args.verbose:
            logging.basicConfig(level=logging.DEBUG)

        self.metadata = self.args_to_network_metadata(self._args)
        self.check_network_metadata_is_supported(self.metadata)

    def add_args(self, parser) -> argparse.ArgumentParser:
        general_group = parser.add_argument_group("general")
        general_group.add_argument(
            "--verbose", help="Display verbose logs.", action="store_true"
        )
        general_group.add_argument(
            "--cleanup", help="Cleans up user-specified workspace. Can not be cleaned if external files exist in workspace.",
            action="store_false"
        )

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
            raise NotImplementedError(
                "The following network config is not yet supported by our scripts: {}".format(
                    metadata
                )
            )

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
        keep_onnx_model: bool,
        keep_pytorch_model: bool,
        timing_profile: TimingProfile,
    ) -> List[NetworkResult]:
        pass

    def __call__(self):
        super().__call__()

        checkpoint = NNSemanticCheckpoint(
            "checkpoint.toml",
            network_name=self.config.network_name,
            metadata=self.metadata,
        )
        network_result = self.run_framework(
            self.metadata,
            list(checkpoint.inputs()),
            self._args.working_dir,
            self._args.cleanup,
            self._args.cleanup,
            TimingProfile(
                iterations=int(self._args.iterations),
                number=int(self._args.number),
                warmup=int(self._args.warmup),
            ),
        )

        return network_result

    def add_args(self, parser) -> argparse.ArgumentParser:
        super().add_args(parser)


class PolygraphyCommand(NetworkCommand):
    """Base class that is associated with Polygraphy related scripts."""

    def __init__(
        self,
        network_config: NNConfig,
        description: str,
        frameworks_cmd: FrameworkCommand,
    ):
        super().__init__(network_config, description)
        # Should be set by
        self.frameworks_cmd = frameworks_cmd()

    @abstractmethod
    def run_polygraphy(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Tuple[NetworkModel],
        network_input: List[str],
        working_directory: str,
        keep_trt_engine: bool,
        keep_onnx_model: bool,
        keep_torch_model: bool,
        timing_profile: TimingProfile,
    ) -> List[NetworkResult]:
        pass

    def __call__(self):
        super().__call__()
        onnx_fpaths = self.args_to_network_models(self._args)

        checkpoint = NNSemanticCheckpoint(
            "checkpoint.toml",
            network_name=self.config.network_name,
            metadata=self.metadata,
        )
        network_result = self.run_polygraphy(
            self.metadata,
            onnx_fpaths,
            list(checkpoint.inputs()),
            self._args.working_dir,
            self._args.cleanup,
            self._args.cleanup,
            self._args.cleanup,
            TimingProfile(
                iterations=int(self._args.iterations),
                number=int(self._args.number),
                warmup=int(self._args.warmup),
            ),
        )

        # compare each result to a known output
        return network_result

    @abstractmethod
    def args_to_network_models(self, args) -> Tuple[NetworkModel]:
        """
        Converts argparse arguments into a list of valid NetworkModel fpaths. Specifically for ONNX.
        Invokes conversion scripts if not.
        Return:
            List[NetworkModel]: List of network model names.
        """


class TRTCommand(NetworkCommand):
    """Base class that is associated with TensorRT related scripts."""

    @abstractmethod
    def run_tensorrt() -> List[NetworkResult]:
        # TODO: Still need to define interface
        pass

    def __call__(self):
        super().__call__()
