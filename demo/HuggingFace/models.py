"""
File for containing ONNX model abstraction. Useful for generating models.
"""

# std
from os import remove
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple
from collections import OrderedDict
from shutil import copytree, rmtree
from logging import debug

# torch
from torch import load, save
from torch.nn import Module


class ModelConverter:
    """Abstract class for converting one model format to another."""

    def __init__(self, onnx_class, torch_class):
        self.onnx_class = onnx_class
        self.torch_class = torch_class

    def torch_to_onnx(self, output_fpath: str, model: Module):
        """
        Converts a torch.Model into an ONNX model on disk specified at output_fpath.

        Arg:
            output_fpath (str): File location of the generated ONNX file.

        Returns:
            ONNXModel: Newly generated ONNXModel
        """
        raise NotImplementedError(
            "Current model does not support exporting to ONNX model."
        )

    def onnx_to_torch(self, output_fpath: str, input_fpath: str):
        """
        Converts ONNX file into torch.Model which is written to disk.

        Arg:
            output_fpath (str): File location of the generated ONNX file.

        Returns:
            TorchModel: Newly generated TorchModel
        """
        raise NotImplementedError(
            "Current model does not support exporting to torch model."
        )


class Dims:
    BATCH = "BATCH_DIM"
    SEQUENCE = "SEQUENCE_DIM"

    def __init__(self, encoding: OrderedDict[str, Tuple[Union[int, str]]]):
        self.encoding = encoding

    def get_dims(self):
        """
        Returns the encoding dimensions.

        Return:
            Dict[str, Union[int, str]]: Returns dimensional encoding. Example: {'input_ids': (1, SEQUENCE_DIM)}
        """
        return self.encoding

    def get_names(self) -> Tuple[str]:
        return tuple(self.encoding.keys())

    def get_lengths(self) -> Tuple[Union[int, str]]:
        return tuple(self.encoding.values())

    def get_dims_with_substitute(self, subs: OrderedDict[str, Tuple[int]]):
        """
        Subtitutes values used in encoding with valid numbers.

        Args:
            subs (Dict[str, int]): Dictionary encoding to disambiguate values. Example: {BATCH_DIM: 1, SEQUENCE_DIM: 128}

        Return:
            Dict[str, int]: Dictionary encoding of dimensions with values substituted:
                            {'input_ids': (1, SEQUENCE_DIM)} => {'input_ids': (1, 512)}
        """
        result = {}
        assert all(isinstance(v, int) for v in result.values())
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


class NNModel(metaclass=ABCMeta):
    """
    Model abstraction. Allows for loading model as various formats.
    The class assumes models live on the disk in order to reduce complexity of model loading into memory.
    The class guarantees that once export functions are called, models exist on the disk for other
    code to parse or use in other libraries.
    """

    def __init__(self, default_converter: ModelConverter = None):
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

    @abstractmethod
    def as_torch_model(self, output_fpath: str, converter: ModelConverter = None):
        """
        Converts ONNX file into torch.Model which is written to disk.
        Uses provided converter to convert object or default_convert is used instead if available.

        Arg:
            output_fpath (str): File location of the generated ONNX file.

        Returns:
            TorchModel: Newly generated TorchModel
        """

    @abstractmethod
    def as_onnx_model(self, output_fpath: str, converter: ModelConverter = None):
        """
        Converts current model into an ONNX model.
        Uses provided converter to convert object or default_convert is used instead if available.

        Args:
            output_fpath (str): File location of the generated ONNX file.

        Returns:
            ONNXModel: Newly generated ONNXModel
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Cleans up any saved models or loaded models from memory."""

    @staticmethod
    @abstractmethod
    def get_input_dims(self) -> Dims:
        """
        Returns the required input id format used by the model
        Arg:
            None

        Return:
            Dims: obtains the dimensions from input.
        """

    @staticmethod
    @abstractmethod
    def get_output_dims(self) -> Dims:
        """
        Returns the required output id format, used by the model.
        Arg:
            None

        Return:
            Dims: the dimensions from output.
        """


class TorchModel(NNModel):

    def __init__(
        self, model: Union[str, Module], default_converter: ModelConverter = None
    ):
        """
        Since torch functions often allow for models to either be from disk as fpath or from a loaded object,
        we provide a similar option here. Arguments can either be a path on disk or from model itself.

        Args:
            model (Union[str, torch.Model]): Location of the model as fpath OR loaded torch.Model object.
        """
        super().__init__(default_converter)

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

    def as_onnx_model(self, output_fpath: str, converter: ModelConverter = None):
        """Converts the torch model into an onnx model."""
        onnx_model = None
        if converter:
            onnx_model = converter().torch_to_onnx(output_fpath, self.load_model())
        else:
            onnx_model = self.default_converter.torch_to_onnx(
                output_fpath, self.load_model()
            )

        return onnx_model

    def as_torch_model(self, output_fpath: str, converter: ModelConverter = None):
        """Since the model is already a torch model, forces a save to specified folder and returns new TorchModel object from that file location."""
        if self.is_loaded:
            save(self.model, output_fpath)
        else:
            copytree(self.fpath, output_fpath)

        torch_model = None
        if converter:
            torch_model = converter().torch_class(output_fpath)
        else:
            torch_model = self.default_converter.torch_class(output_fpath)

        return torch_model

    def cleanup(self):
        if self.model:
            debug("Freeing model from memory: {}".format(self.model))
            del self.model

        if self.fpath:
            debug("Removing saved torch model from location: {}".format(self.fpath))
            rmtree(self.fpath)


class ONNXModel(NNModel):

    def __init__(self, model: str, default_converter: ModelConverter = None):
        """
        Keeps track of ONNX model file. Does not support loading into memory. Only reads and writes to disk.

        Args:
            model (str): Location of the model as fpath OR loaded torch.Model object.
        """
        super().__init__(default_converter)
        self.fpath = model

    def as_onnx_model(self, output_fpath: str, converter: ModelConverter = None):
        """Since the model is already a torch model, forces a save to specified folder and returns new ONNXModel object from that file location."""
        copytree(self.fpath, output_fpath)

        onnx_model = None
        if converter:
            onnx_model = converter().onnx_class(output_fpath)
        else:
            onnx_model = self.default_converter.onnx_class(output_fpath)

        return onnx_model

    def as_torch_model(self, output_fpath: str, converter: ModelConverter = None):
        """Converts the onnx model into an torch model."""
        torch_model = None
        if converter:
            torch_model = converter().onnx_to_torch(output_fpath, self.fpath)
        else:
            torch_model = self.default_converter.onnx_to_torch(output_fpath, self.fpath)

        return torch_model

    def cleanup(self):
        debug("Removing saved ONNX model from location: {}".format(self.fpath))
        remove(self.fpath)


class NullConverter(ModelConverter):
    def __init__(self):
        super().__init__(ONNXModel, TorchModel)
