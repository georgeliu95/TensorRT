"""
Contains logic that captures GPT2 HuggingFace models into ONNX models.
"""
# torch
import torch

from torch.nn import Module

# # huggingface
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import GPT2Tokenizer

# TRT-HuggingFace
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from networks import NetworkMetadata
from models import TRTEngineFile, TorchModelFile, ONNXModelFile, ModelFileConverter

class GPT2TorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of GPT2 with LM head.
        """

        def __init__(self, transformer, lm_head, config):
            super().__init__()
            self.transformer = transformer
            self.lm_head = lm_head
            self.config = config

        def prepare_inputs_for_generation(self, input_ids, **kwargs):  
            # Todo (@pchadha): add position_ids, token_type_ids support
            return {
                "input_ids": input_ids,
            }

        def forward(self, input_ids, **kwargs):
            transformer_outputs = self.transformer(
                input_ids
            )        
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)
            return CausalLMOutputWithCrossAttentions(logits=lm_logits)
        
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata):
        super().__init__(model, GPT2Converter, network_metadata)

class GPT2ONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, GPT2Converter, network_metadata)


# TRT Engine File Encoding #
class GPT2TRTEngine(TRTEngineFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, GPT2Converter, network_metadata)

# Converters
class GPT2Converter(ModelFileConverter):
    def __init__(self):
        super().__init__(GPT2TorchFile, GPT2ONNXFile, GPT2TRTEngine)

    def torch_to_onnx(self, output_fpath: str, model: Module, network_metadata: NetworkMetadata):
        """
        Exports a GPT2LMHead model to ONNX.

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            T5DecoderONNXFile: ONNX decoder object.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(network_metadata.variant)
        input_ids = torch.tensor(
            [tokenizer.encode("Here is some text to encode Hello World", add_special_tokens=True)])

        gpt2_model = GPT2TorchFile.TorchModule(model.transformer, model.lm_head, model.config)
        inputs = GPT2ModelTRTConfig.get_input_dims(network_metadata)[GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME]
        outputs = GPT2ModelTRTConfig.get_output_dims(network_metadata)[GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME]

        # Exports to ONNX
        torch.onnx._export(
            gpt2_model,
            input_ids,
            output_fpath,
            opset_version=12,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=False,
        )
        return GPT2ONNXFile(output_fpath, network_metadata)

