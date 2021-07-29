"""
File for containing model file abstraction. Useful for generating models.
"""

# std
import os
import logging

from abc import ABCMeta, abstractmethod
from typing import Union, Tuple
from collections import OrderedDict
from shutil import copytree, rmtree
from logging import debug

# polygraphy
from polygraphy.backend.trt import (
    network_from_onnx_path,
    Profile,
    engine_from_network,
    save_engine,
)

from polygraphy.backend.trt import TrtRunner, CreateConfig, engine_from_bytes
from polygraphy.backend.common import bytes_from_path
from polygraphy.logger import G_LOGGER

# torch
from torch import load, save
from torch.nn import Module

# TRT-HuggingFace
from networks import NetworkMetadata


class ModelFileConverter:
    """Abstract class for converting one model format to another."""

    def __init__(self, onnx_class, torch_class, trt_engine_class):
        self.onnx_class = onnx_class
        self.torch_class = torch_class
        self.trt_engine_class = trt_engine_class

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Converts a torch.Model into an ONNX model on disk specified at output_fpath.

        Arg:
            output_fpath (str): File location of the generated ONNX file.

        Returns:
            ONNXModelFile: Newly generated ONNXModelFile
        """
        raise NotImplementedError(
            "Current model does not support exporting to ONNX model."
        )

    def onnx_to_torch(
        self, output_fpath: str, input_fpath: str, network_metadata: NetworkMetadata
    ):
        """
        Converts ONNX file into torch.Model which is written to disk.

        Arg:
            output_fpath (str): File location of the generated ONNX file.

        Returns:
            TorchModelFile: Newly generated TorchModelFile
        """
        raise NotImplementedError(
            "Current model does not support exporting to torch model."
        )


class Dims:
    """Helper class for interfacing dimension constructs with Polygraphy and PyTorch."""

    BATCH = "BATCH_DIM"
    SEQUENCE = "SEQUENCE_DIM"

    def __init__(self, encoding: OrderedDict):
        self.encoding = encoding

    def get_dims(self):
        """
        Returns the encoding dimensions.

        Return:
            OrderedDict[str, Union[int, str]]: Returns dimensional encoding. Example: {'input_ids': (1, SEQUENCE_DIM)}
        """
        return self.encoding

    def get_names(self) -> Tuple[str]:
        return tuple(self.encoding.keys())

    def get_lengths(self) -> Tuple[Union[int, str]]:
        return tuple(self.encoding.values())

    def get_dims_with_substitute(self, subs: OrderedDict):
        """
        Subtitutes values used in encoding with valid numbers.

        Args:
            subs (OrderedDict[str, int]): Dictionary encoding to disambiguate values. Example: {BATCH_DIM: 1, SEQUENCE_DIM: 128}

        Return:
            OrderedDict[str, int]: Dictionary encoding of dimensions with values substituted:
                            {'input_ids': (1, SEQUENCE_DIM)} => {'input_ids': (1, 512)}
        """
        result = {}
        assert all(isinstance(v, int) for v in subs.values())
        return result

    def get_torch_dynamic_axis_encoding(self) -> dict:
        """
        Returns a Pytorch "dynamic_axes" encoding for onnx.export.

        Returns:
            dict: Returns a 'dynamic' index with corresponding names according to:
                https://pytorch.org/docs/stable/onnx.html
        """

        dynamic_axes = {}
        for k, v in self.encoding.items():
            encodings = []
            for e in v:
                if e == self.BATCH:
                    encodings.append("batch")
                elif e == self.SEQUENCE:
                    encodings.append("sequence")
            dynamic_axes[k] = {c: v for c, v in enumerate(encodings)}

        return dynamic_axes


class NNModelFile(metaclass=ABCMeta):
    """
    Model abstraction. Allows for loading model as various formats.
    The class assumes models live on the disk in order to reduce complexity of model loading into memory.
    The class guarantees that once export functions are called, models exist on the disk for other
    code to parse or use in other libraries.
    """

    def __init__(
        self,
        default_converter: ModelFileConverter = None,
        network_metadata: NetworkMetadata = None,
    ):
        """
        Since torch functions often allow for models to either be from disk as fpath or from a loaded object,
        we provide a similar option here. Arguments can either be a path on disk or from model itself.

        Args:
            model (Union[str, torch.Model]): Location of the model as fpath OR loaded torch.Model object.
        """
        if default_converter is not None:
            self.default_converter = default_converter()
        else:
            self.default_converter = NullConverter()

        self.network_metadata = network_metadata

    def as_torch_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts ONNX file into torch.Model which is written to disk.
        Uses provided converter to convert object or default_convert is used instead if available.

        Arg:
            output_fpath (str): File location of the generated torch file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.

        Returns:
            TorchModelFile: Newly generated TorchModelFile
        """
        raise NotImplementedError("Current model does not support exporting to pytorch model.")

    def as_onnx_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts current model into an ONNX model.
        Uses provided converter to convert object or default_convert is used instead if available.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.

        Returns:
            ONNXModelFile: Newly generated ONNXModelFile
        """
        raise NotImplementedError("Current model does not support exporting to onnx model.")

    def as_trt_engine(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts current model into an TRT engine.
        Uses provided converter to convert object or default_convert is used instead if available.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.

        Returns:
            TRTEngineFile: Newly generated ONNXModelFile
        """
        raise NotImplementedError("Current model does not support exporting to trt engine.")


    @abstractmethod
    def cleanup(self) -> None:
        """Cleans up any saved models or loaded models from memory."""


class TorchModelFile(NNModelFile):
    def __init__(
        self,
        model: Union[str, Module],
        default_converter: ModelFileConverter = None,
        network_metadata: NetworkMetadata = None,
    ):
        """
        Since torch functions often allow for models to either be from disk as fpath or from a loaded object,
        we provide a similar option here. Arguments can either be a path on disk or from model itself.

        Args:
            model (Union[str, torch.Model]): Location of the model as fpath OR loaded torch.Model object.
        """
        super().__init__(default_converter, network_metadata)

        if isinstance(model, Module):
            self.is_loaded = True
            self.fpath = None
            self.model = model
        else:
            self.is_loaded = False
            self.fpath = model
            self.model = None

    def load_model(self) -> Module:
        """
        Loads the model from disk if isn't already loaded.
        Does not attempt to load if given model is already loaded and instead returns original instance.
        Use as_torch_model() instead to always guarantee a new instance and location on disk.

        Args:
            None

        Returns:
            torch.Model: Loaded torch model.
        """
        if self.is_loaded:
            return self.model

        return load(self.fpath)

    def as_onnx_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts the torch model into an onnx model.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
        Return:
            (converter.onnx_class): Returns a converted instance of ONNXModelFile.
        """
        converter = self.default_converter if converter is None else converter()
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.onnx_class(output_fpath, self.network_metadata)

        return converter.torch_to_onnx(
            output_fpath, self.load_model(), self.network_metadata
        )

    def as_torch_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Since the model is already a torch model, forces a save to specified folder and returns new TorchModelFile object from that file location.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
        Return:
            (converter.torch_class): Returns a converted instance of TorchModelFile.
        """
        converter = self.default_converter if converter is None else converter()
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.torch_class(output_fpath, self.network_metadata)

        if self.is_loaded:
            save(self.model, output_fpath)
        else:
            copytree(self.fpath, output_fpath)

        return converter.torch_class(output_fpath, self.network_metadata)

    def cleanup(self) -> None:
        if self.model:
            debug("Freeing model from memory: {}".format(self.model))
            del self.model

        if self.fpath:
            debug("Removing saved torch model from location: {}".format(self.fpath))
            rmtree(self.fpath)


class ONNXModelFile(NNModelFile):
    def __init__(
        self,
        model: str,
        default_converter: ModelFileConverter = None,
        network_metadata: NetworkMetadata = None,
    ):
        """
        Keeps track of ONNX model file. Does not support loading into memory. Only reads and writes to disk.

        Args:
            model (str): Location of the model as fpath OR loaded torch.Model object.
        """
        super().__init__(default_converter, network_metadata)
        self.fpath = model

    def as_onnx_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Since the model is already a onnx model, forces a save to specified folder and returns new ONNXModelFile object from that file location.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.

        Return:
            (converter.onnx_class): Returns a converted instance of ONNXModelFile.
        """
        converter = self.default_converter if converter is None else converter()
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.onnx_class(output_fpath, self.network_metadata)
        else:
            copytree(self.fpath, output_fpath)

        return converter.onnx_class(output_fpath, self.network_metadata)

    def as_torch_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts the onnx model into an torch model.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
        Return:
            (converter.torch_class): Returns a converted instance of TorchModelFile.
        """
        converter = self.default_converter if converter is None else converter()
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.torch_class(output_fpath, self.network_metadata)

        return converter.onnx_to_torch(output_fpath, self.fpath, self.network_metadata)

    def cleanup(self) -> None:
        debug("Removing saved ONNX model from location: {}".format(self.fpath))
        os.remove(self.fpath)

    def as_trt_engine(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False
    ):
        """
        Converts the onnx model into an trt engine.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
        Return:
            (converter.trt_engine_class): Returns a converted instance of TRTEngineFile.
        """
        converter = self.default_converter if converter is None else converter()
        # TODO: Need to check if the old engine file is compatible with current setting
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.trt_engine_class(output_fpath, self.network_metadata)

        result = converter.trt_engine_class(output_fpath, self.network_metadata)
        self.trt_inference_config = CreateConfig(
            fp16=self.network_metadata.precision.fp16,
            max_workspace_size=result.DEFAULT_TRT_WORKSPACE_MB * 1024 * 1024,
            profiles=result.get_dynamic_shape_profiles(),
        )

        g_logger_verbosity = G_LOGGER.EXTRA_VERBOSE if logging.root.level == logging.DEBUG else G_LOGGER.WARNING
        with G_LOGGER.verbosity(g_logger_verbosity):
            self.trt_engine = engine_from_network(
                network_from_onnx_path(self.fpath), config=self.trt_inference_config
            )
            save_engine(self.trt_engine, output_fpath)

        return result

class TRTEngineFile(NNModelFile):
    DEFAULT_TRT_WORKSPACE_MB = 1024

    @abstractmethod
    def get_dynamic_shape_profiles(self):
        pass

    def __init__(self, model: str, default_converter: ModelFileConverter = None, network_metadata: NetworkMetadata = None):
        super().__init__(default_converter, network_metadata)
        self.fpath = model

        if os.path.exists(self.fpath):
            # Engine already exists, do nothing
            return

    def cleanup(self) -> None:
        debug("Removing saved engine model from location: {}".format(self.fpath))

class NullConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(ONNXModelFile, TorchModelFile, TRTEngineFile)
