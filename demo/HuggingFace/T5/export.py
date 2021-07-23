"""
Contains logic that captures T5 HuggingFace models into ONNX models.
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py
"""
# std
from collections import OrderedDict

# torch
import torch
from transformers.utils.dummy_pt_objects import (
    PreTrainedModel,
)

from models import TorchModelFile, ONNXModelFile, ModelFileConverter, Dims
from torch.nn import Module

# huggingface
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput


class T5DecoderTorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of T5 Decoder without support for loss.
        Decoder with lm-head attached.
        """

        def __init__(self, decoder, lm_head, config):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.config = config

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {
                "input_ids": input_ids,
                "encoder_hidden_states": kwargs["encoder_hidden_states"],
            }

        def forward(self, input_ids, encoder_hidden_states, **kwargs):
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs
            )

            # self.config.d_model ** -0.5 for rescaling output on vocab.
            # as seen in https://huggingface.co/transformers/_modules/transformers/models/t5/modeling_t5.html#T5ForConditionalGeneration
            sequence_output = decoder_outputs[0] * self.config.d_model ** -0.5
            logits = self.lm_head(sequence_output)
            if not kwargs.get("return_dict", False):
                return (logits,) + decoder_outputs[1:]

            return Seq2SeqLMOutput(logits=logits)

    def __init__(self, model):
        TorchModelFile.__init__(self, model, T5DecoderConverter)

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


class T5EncoderTorchFile(TorchModelFile, GenerationMixin, PreTrainedModel):
    """Creation of a class to output only the last hidden state from the encoder."""

    class TorchModule(Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, *input, **kwargs):
            return self.encoder(*input, **kwargs)[0]

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model):
        TorchModelFile.__init__(self, model, T5EncoderConverter)

    @staticmethod
    def get_input_dims():
        """Returns the input ids for T5."""
        return Dims(OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)}))

    @staticmethod
    def get_output_dims():
        return Dims(OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE)}))


class T5EncoderONNXFile(ONNXModelFile):
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


class T5DecoderONNXFile(ONNXModelFile):
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
class T5DecoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5DecoderTorchFile, T5DecoderONNXFile)

    def torch_to_onnx(self, output_fpath: str, model: Module):
        """
        Exports a given huggingface T5 to decoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            T5DecoderONNXFile: ONNX decoder object.
        """

        input_ids = torch.tensor([[42] * 10])
        # Exporting the decoder requires a basic instance of the encoder
        # Create on temporarily
        simplified_encoder = T5EncoderTorchFile.TorchModule(model.encoder)
        # Exports to ONNX
        decoder_with_lm_head = T5DecoderTorchFile.TorchModule(
            model.decoder, model.lm_head, model.config
        )
        inputs = T5DecoderTorchFile.get_input_dims()
        outputs = T5DecoderTorchFile.get_output_dims()
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

        return T5DecoderONNXFile(output_fpath)


class T5EncoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5EncoderTorchFile, T5EncoderONNXFile)

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
        simplified_encoder = T5EncoderTorchFile.TorchModule(model.encoder)
        inputs = T5EncoderTorchFile.get_input_dims()
        outputs = T5EncoderTorchFile.get_output_dims()
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

        return T5EncoderONNXFile(output_fpath)
