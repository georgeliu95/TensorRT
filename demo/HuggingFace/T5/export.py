"""
Contains logic that captures T5 HuggingFace models into ONNX models.
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py
"""
# std
from collections import OrderedDict

# torch
import torch

from models import TorchModel, ONNXModel, ModelConverter, Dims
from torch.nn import Module


class T5TorchDecoder(TorchModel):
    class TorchModule(Module):
        """Decoder with lm-head attached."""

        def __init__(self, decoder, lm_head, config):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.config = config

        def forward(self, input_ids, encoder_hidden_states):
            # self.config.d_model ** -0.5 is most probably for normalization
            # it was in the original code from ONNX model repo.
            decoder_output = self.decoder(
                input_ids=input_ids, encoder_hidden_states=encoder_hidden_states
            )[0] * (self.config.d_model ** -0.5)
            return self.lm_head(decoder_output)

    def __init__(self, model):
        super().__init__(model, T5DecoderConverter)

    @staticmethod
    def get_input_dims():
        """Returns the input ids for T5."""
        return Dims(
            OrderedDict(
                {
                    "input_ids": (Dims.BATCH, Dims.SEQUENCE),
                    "encoder_hidden_states": (Dims.BATCH, Dims.SEQUENCE),
                }
            )
        )

    @staticmethod
    def get_output_dims():
        return Dims(OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE)}))


class T5TorchEncoder(TorchModel):
    class TorchModule(Module):
        """Creation of a class to output only the last hidden state from the encoder."""

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, *input, **kwargs):
            return self.encoder(*input, **kwargs)[0]

    def __init__(self, model):
        super().__init__(model, T5EncoderConverter)

    @staticmethod
    def get_input_dims():
        """Returns the input ids for T5."""
        return Dims(OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)}))

    @staticmethod
    def get_output_dims():
        return Dims(OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE)}))


class T5ONNXEncoder(ONNXModel):
    def __init__(self, model):
        super().__init__(model, T5EncoderConverter)

    @staticmethod
    def get_input_dims():
        """Returns the i3put ids for T5."""
        return Dims(
            OrderedDict(
                {
                    "input_ids": (Dims.BATCH, Dims.SEQUENCE),
                    "encoder_hidden_states": (Dims.BATCH, Dims.SEQUENCE),
                }
            )
        )

    @staticmethod
    def get_output_dims():
        return Dims(OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE)}))


class T5ONNXDecoder(ONNXModel):
    def __init__(self, model):
        super().__init__(model, T5DecoderConverter)

    @staticmethod
    def get_input_dims():
        """Returns the input ids for T5."""
        return Dims(OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)}))

    @staticmethod
    def get_output_dims():
        return Dims(OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE)}))


# Converters
class T5DecoderConverter(ModelConverter):
    def __init__(self):
        super().__init__(T5TorchDecoder, T5ONNXDecoder)

    def torch_to_onnx(self, output_fpath: str, model: Module):
        """
        Exports a given huggingface T5 to decoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            T5ONNXDecoder: ONNX decoder object.
        """

        input_ids = torch.tensor([[42] * 10])
        simplified_encoder = T5TorchEncoder.TorchModule(model.encoder)

        # Exports to ONNX
        decoder_with_lm_head = T5TorchDecoder.TorchModule(
            model.decoder, model.lm_head, model.config
        )
        inputs = T5TorchDecoder.get_input_dims()
        outputs = T5TorchDecoder.get_output_dims()
        torch.onnx.export(
            decoder_with_lm_head,
            (input_ids, simplified_encoder(input_ids)),
            output_fpath,
            export_params=True,
            opset_version=12,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=False,
        )

        return T5ONNXDecoder(output_fpath)


class T5EncoderConverter(ModelConverter):
    def __init__(self):
        super().__init__(T5TorchEncoder, T5ONNXEncoder)

    def torch_to_onnx(self, output_fpath: str, model: Module):
        """
        Exports a given huggingface T5 to encoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            Tuple[str]: Names of generated models
        """
        input_ids = torch.tensor([[42] * 10])
        simplified_encoder = T5TorchEncoder.TorchModule(model.encoder)
        inputs = T5TorchEncoder.get_input_dims()
        outputs = T5TorchEncoder.get_output_dims()
        encoder_hidden_states = Dims(
            OrderedDict({"encoder_hidden_states": (Dims.BATCH, Dims.SEQUENCE)})
        )

        # Exports to ONNX
        torch.onnx._export(
            simplified_encoder,
            input_ids,
            output_fpath,
            export_params=True,
            opset_version=12,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **encoder_hidden_states.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=False,
        )

        return T5ONNXEncoder(output_fpath)
