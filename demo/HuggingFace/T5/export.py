"""
Contains logic that captures T5 HuggingFace models into ONNX models.
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py
"""
# std
from polygraphy.backend.trt import Profile

# torch
import torch
from torch.nn import Module

# huggingface
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

# TRT-HuggingFace
from T5.T5ModelConfig import T5ModelTRTConfig
from networks import NetworkMetadata
from models import TRTEngineFile, TorchModelFile, ONNXModelFile, ModelFileConverter, Dims


# Torch File Encoding #
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

    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)


class T5EncoderTorchFile(TorchModelFile):
    """Creation of a class to output only the last hidden state from the encoder."""

    class TorchModule(Module, GenerationMixin):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, *input, **kwargs):
            return self.encoder(*input, **kwargs)[0]

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)


# ONNX File Encoding #
class T5EncoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)


class T5DecoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)


# TRT Engine File Encoding #
class T5DecoderTRTEngine(TRTEngineFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)

    def get_dynamic_shape_profiles(self):
        max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[
            self.network_metadata.variant
        ]
        profile = Profile()
        profile.add(
            "input_ids",
            min=(1, 256),
            opt=(1, 256),
            max=(1, 256),
        )
        profile.add(
            "encoder_hidden_states",
            min=(1, 256, max_sequence_length),
            opt=(1, 256, max_sequence_length),
            max=(1, 256, max_sequence_length),
        )
        return [profile]

class T5EncoderTRTEngine(TRTEngineFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)

    def get_dynamic_shape_profiles(self):
        max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[
            self.network_metadata.variant
        ]
        return [
            Profile().add(
                "input_ids",
                min=(1, 1),
                opt=(1, max_sequence_length // 2),
                max=(1, max_sequence_length),
            )
        ]


# Converters #
class T5DecoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5DecoderTorchFile, T5DecoderONNXFile, T5DecoderTRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
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
        # Create one temporarily
        simplified_encoder = T5EncoderTorchFile.TorchModule(model.encoder)
        # Exports to ONNX
        decoder_with_lm_head = T5DecoderTorchFile.TorchModule(
            model.decoder, model.lm_head, model.config
        )
        inputs = T5ModelTRTConfig.get_input_dims(network_metadata)["decoder"]
        outputs = T5ModelTRTConfig.get_output_dims(network_metadata)["decoder"]

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

        return T5DecoderONNXFile(output_fpath, network_metadata)


class T5EncoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5EncoderTorchFile, T5EncoderONNXFile, T5EncoderTRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
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
        inputs = T5ModelTRTConfig.get_input_dims(network_metadata)["encoder"]
        outputs = T5ModelTRTConfig.get_output_dims(network_metadata)["encoder"]

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
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=False,
        )

        return T5EncoderONNXFile(output_fpath, network_metadata)
