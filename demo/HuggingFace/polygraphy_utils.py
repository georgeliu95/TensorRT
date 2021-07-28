"""Utilities related to Polygraphy"""

# std
import logging

# TRT-HuggingFace
from polygraphy.backend.trt import engine_from_bytes, TrtRunner
from polygraphy.backend.common import bytes_from_path
from polygraphy.logger import G_LOGGER
from networks import NetworkMetadata

class TRTRunner:
    """TRT implemented network interface that can be used to measure inference time."""

    def __init__(self, engine_fpath: str, network_metadata: NetworkMetadata):
        self.network_metadata = network_metadata

        self.trt_engine = engine_from_bytes(bytes_from_path(engine_fpath))
        self.trt_context = TrtRunner(self.trt_engine.create_execution_context())
        self.trt_context.activate()

    def __call__(self, *args, **kwargs):
        # hook polygraphy verbosity for inference
        g_logger_verbosity = G_LOGGER.EXTRA_VERBOSE if logging.root.level == logging.DEBUG else G_LOGGER.WARNING
        with G_LOGGER.verbosity(g_logger_verbosity):
            return self.forward(*args, **kwargs)

    def release(self):
        self.trt_context.deactivate()
