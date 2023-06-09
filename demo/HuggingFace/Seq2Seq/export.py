#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Contains logic that captures HuggingFace models into ONNX models.
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py
"""

# tensorrt
import tensorrt as trt

# torch
import torch
from torch.nn import Module

# Huggingface
from transformers import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from NNDF.networks import NetworkMetadata
from NNDF.logger import G_LOGGER
from NNDF.models import (
    TRTEngineFile,
    TorchModelFile,
    ONNXModelFile,
    ModelFileConverter,
)

from NNDF.networks import NetworkMetadata

# In gpt-j, torch.repeat_interleave will export SplitToSequence with subgraph. Use transformers==4.27.4 as a patch to avoid it
# https://github.com/pytorch/pytorch/pull/100575 is merged and is expected to fix this issue in the next PyTorch release.
from transformers.models.gptj.modeling_gptj import rotate_every_two
import unittest.mock as mock

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[1]
    m = m.reshape(1,-1,1)  # flatten the matrix
    m = m.repeat(1,1,2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, 1, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, _sin, _cos):
    sin = duplicate_interleave(_sin)[None,:,:,:]
    cos = duplicate_interleave(_cos)[None,:,:,:]
    return (x * cos) + (rotate_every_two(x) * sin)

OPSET = 17
TRAINING_MODE = torch.onnx.TrainingMode.EVAL
CONSTANT_FOLDING = True

# Encoder File Encoding #
class EncoderTorchFile(TorchModelFile):

    class TorchModule(Module):
        def __init__(self, model: Module):
            super().__init__()
            self.encoder = model.get_encoder()
            self.device = model.device

        def forward(self, *input, **kwargs):
            encoder_hidden_states = self.encoder(*input, **kwargs)[0]
            return BaseModelOutput(last_hidden_state = encoder_hidden_states)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = EncoderConverter

        super().__init__(model, default_converter, network_metadata)

class EncoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = EncoderConverter

        super().__init__(model, default_converter, network_metadata)

class EncoderTRTEngine(TRTEngineFile):

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = EncoderConverter

        super().__init__(model, default_converter, network_metadata)

    def get_network_definition(self, network_definition):
        if self.network_metadata.precision.fp16:
            for i in range(network_definition[1].num_inputs):
                t = network_definition[1].get_input(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

            for i in range(network_definition[1].num_outputs):
                t = network_definition[1].get_output(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

        return network_definition

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16

class EncoderConverter(ModelFileConverter):
    def __init__(self,
        torch_class=EncoderTorchFile,
        onnx_class=EncoderONNXFile,
        trt_engine_class=EncoderTRTEngine
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)


    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata, config
    ):
        """
        Exports a given huggingface Seq2Seq to encoder architecture only.
        Inspired by https://github.com/onnx/models/blob/main/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_fpath (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            Tuple[str]: Names of generated models
        """
        device = model.device
        input_ids = torch.tensor([[42] * 10]).to(device)
        simplified_encoder = self.torch_class.TorchModule(model)
        inputs = config.get_input_dims()[config.NETWORK_ENCODER_SEGMENT_NAME]
        outputs = config.get_output_dims()[config.NETWORK_ENCODER_SEGMENT_NAME]

        # Exports to ONNX
        torch.onnx.export(
            simplified_encoder,
            input_ids,
            output_fpath,
            do_constant_folding=True,
            opset_version=OPSET,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=torch.onnx.TrainingMode.EVAL,
        )

        if network_metadata.precision.fp16:
            self.post_process_onnx(output_fpath)

        return self.onnx_class(model = output_fpath, network_metadata = network_metadata)

# Decoder File Encoding #
class DecoderTorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of Seq2Seq Decoder without support for loss.
        Decoder with lm-head attached.
        """

        def __init__(self, model: Module):
            super().__init__()
            self.is_encoder_decoder = model.config.is_encoder_decoder
            if self.is_encoder_decoder:
                self.decoder = model.get_decoder()
            else:
                if hasattr(model, "transformer"):
                    self.decoder = model.transformer
                elif hasattr(model, "model"):
                    self.decoder = model.model
                elif hasattr(model, "gpt_neox"): # For gpt-neox-20b
                    self.decoder = model.gpt_neox
                else:
                    raise RuntimeError("Model does not have either transformer or model attr.")

            if hasattr(model, "lm_head"):
                self.lm_head = model.lm_head
            elif hasattr(model, "embed_out"):
                self.lm_head = model.embed_out
            else:
                raise RuntimeError("Model does not have a lm_head attr")

            self.config = model.config

            # Required for decoder generation for HuggingFace
            if hasattr(model, "main_input_name"):
                self.main_input_name = model.main_input_name
            if hasattr(model, "generation_config"):
                self.generation_config = model.generation_config

            self.device = model.device

        def can_generate(self):
            return True

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values = None,
            use_cache = None,
            encoder_outputs = None,
        ):

            if past_key_values is not None:
                input_ids = input_ids[:, -1:]

            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "encoder_outputs": encoder_outputs,
                "use_cache": use_cache,
            }

        @mock.patch("transformers.models.gptj.modeling_gptj.apply_rotary_pos_emb", apply_rotary_pos_emb)
        def forward(
            self,
            input_ids,
            encoder_outputs = None,
            past_key_values = None,
            use_cache = None,
            **kwargs,
        ):
            encoder_args = {}
            if self.config.is_encoder_decoder:
                encoder_args["encoder_hidden_states"] = encoder_outputs.last_hidden_state

            decoder_outputs = self.decoder(
                input_ids=input_ids,
                use_cache=use_cache,
                past_key_values=past_key_values,
                **encoder_args,
                **kwargs
            )

            sequence_output = decoder_outputs[0]
            logits = self.lm_head(sequence_output) if self.lm_head is not None else sequence_output
            past_key_values = decoder_outputs[1] if use_cache else None

            return Seq2SeqLMOutput(
                logits=logits,
                past_key_values=past_key_values
            )

        def _reorder_cache(self, past_key_values, beam_idx):
            # if decoder past is not included in output
            # speedy decoding is disabled and no need to reorder
            if past_key_values is None:
                G_LOGGER.warning("You might want to consider setting `use_cache=True` to speed up decoding")
                return past_key_values

            reordered_decoder_past = ()
            for layer_past_states in past_key_values:
                # get the correct batch idx from layer past batch dim
                # batch dim of `past` is at 2nd position
                reordered_layer_past_states = ()
                for layer_past_state in layer_past_states:
                    # need to set correct `past` for each of the four key / value states
                    reordered_layer_past_states = reordered_layer_past_states + (
                        layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                    )

                assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
                assert len(reordered_layer_past_states) == len(layer_past_states)

                reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
            return reordered_decoder_past

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = DecoderConverter
        super().__init__(model, default_converter, network_metadata)

class DecoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = DecoderConverter
        super().__init__(model, default_converter, network_metadata)

class DecoderTRTEngine(TRTEngineFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = DecoderConverter
        super().__init__(model, default_converter, network_metadata)

    def get_network_definition(self, network_definition):
        if self.network_metadata.precision.fp16:
            for i in range(network_definition[1].num_inputs):
                t = network_definition[1].get_input(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

            for i in range(network_definition[1].num_outputs):
                t = network_definition[1].get_output(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

        return self.set_layer_precisions(network_definition)

    def set_layer_precisions(self, network_definition):
        """
        Users could customize this function for different model if mixed precision is needed.
        """
        return network_definition

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16

class DecoderConverter(ModelFileConverter):
    def __init__(self,
        torch_class=DecoderTorchFile,
        onnx_class=DecoderONNXFile,
        trt_engine_class=DecoderTRTEngine,
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata, config
    ):
        """
        Exports a given huggingface Seq2Seq to decoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/Seq2Seq-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            DecoderONNXFile: ONNX decoder object.
        """
        # Adding a device parameter to the class may help
        device = model.device
        input_ids = torch.tensor([[42] * 10]).to(device)
        encoder_hidden_states = None
        if config.is_encoder_decoder:
            simplified_encoder = config.encoder_classes["torch"].TorchModule(model)
            encoder_hidden_states = simplified_encoder(input_ids).last_hidden_state

        # Exports to ONNX
        decoder_with_lm_head = self.torch_class.TorchModule(model)

        # This would require the use of config
        inputs = config.get_input_dims()[config.NETWORK_DECODER_SEGMENT_NAME]
        outputs = config.get_output_dims()[config.NETWORK_DECODER_SEGMENT_NAME]

        # Exports to ONNX
        if not network_metadata.use_cache:
            # This code allows for huggingface compatible torch class to use onnx exporter
            old_forward = decoder_with_lm_head.forward
            def _export_forward(input_ids, encoder_hidden_states):
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
                result = old_forward(input_ids, encoder_outputs=encoder_outputs, use_cache=False)
                return result[0]

            decoder_with_lm_head.forward = _export_forward
            torch.onnx.export(
                decoder_with_lm_head,
                (input_ids, encoder_hidden_states),
                output_fpath,
                do_constant_folding=True,
                opset_version=OPSET,
                input_names=inputs.get_names(),
                output_names=outputs.get_names(),
                dynamic_axes={
                    **inputs.get_torch_dynamic_axis_encoding(),
                    **outputs.get_torch_dynamic_axis_encoding(),
                },
                training=TRAINING_MODE,
            )
        else:
            decoder_input_ids = input_ids[:,-1:]
            decoder_outputs = decoder_with_lm_head.forward(
                input_ids=decoder_input_ids,
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
                use_cache=True,
                past_key_values=None
            ) # decoder output at t-1 step (logits, past_key_values from 0 to t-1)

            past_key_values = decoder_outputs[1]

            old_forward = decoder_with_lm_head.forward

            def _export_forward(input_ids, encoder_hidden_states, past_key_values):
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
                decoder_outputs = old_forward(input_ids, encoder_outputs=encoder_outputs, past_key_values=past_key_values, use_cache=True)
                past_key_values = ()
                past_key_values_output = decoder_outputs[1]
                for layer_past_states in past_key_values_output:
                    past_key_values = past_key_values + (layer_past_states[:2],)

                logits = decoder_outputs.logits
                if encoder_hidden_states is not None:
                    logits = logits.view(encoder_hidden_states.shape[0], logits.shape[1], logits.shape[2])

                return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

            decoder_with_lm_head.forward = _export_forward

            torch.onnx.export(
                decoder_with_lm_head,
                (decoder_input_ids, encoder_hidden_states, past_key_values),
                output_fpath,
                do_constant_folding=True,
                opset_version=OPSET,
                input_names=inputs.get_names(),
                output_names=outputs.get_names(),
                dynamic_axes={
                    **inputs.get_torch_dynamic_axis_encoding(),
                    **outputs.get_torch_dynamic_axis_encoding(),
                },
                training=TRAINING_MODE,
            )

        if network_metadata.precision.fp16:
            self.post_process_onnx(output_fpath)

        return self.onnx_class(model = output_fpath, network_metadata = network_metadata)

# Cross Attention Cache Generator File Encoding
class CrossAttnCacheGeneratorTorchFile(TorchModelFile):
    class TorchModule(Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, encoder_hidden_states):
            raise RuntimeError("You are running use_cache = True with TRT in a encoder_decoder model. \
                               Please write a cross attentoin generator class.")

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = CrossAttnCacheGeneratorConverter

        super().__init__(model, default_converter, network_metadata)

class CrossAttnCacheGeneratorONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = CrossAttnCacheGeneratorConverter

        super().__init__(model, default_converter, network_metadata)

class CrossAttnCacheGeneratorTRTEngine(TRTEngineFile):

    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = CrossAttnCacheGeneratorConverter

        super().__init__(model, default_converter, network_metadata)

    def get_network_definition(self, network_definition):

        if self.network_metadata.precision.fp16:
            for i in range(network_definition[1].num_inputs):
                t = network_definition[1].get_input(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

            for i in range(network_definition[1].num_outputs):
                t = network_definition[1].get_output(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

        return network_definition

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16

class CrossAttnCacheGeneratorConverter(ModelFileConverter):
    def __init__(self,
        torch_class=CrossAttnCacheGeneratorTorchFile,
        onnx_class=CrossAttnCacheGeneratorONNXFile,
        trt_engine_class=CrossAttnCacheGeneratorTRTEngine,
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata, config
    ):
        # This model must be encoder/decoder, so no need to check
        device = model.device
        input_ids = torch.tensor([[42] * 10]).to(device)
        simplified_encoder = config.encoder_classes["torch"].TorchModule(model)
        encoder_hidden_states = simplified_encoder(input_ids).last_hidden_state
        inputs = config.get_input_dims()[config.NETWORK_CROSS_ATTENTION_CACHE_GENERATOR_NAME]
        outputs = config.get_output_dims()[config.NETWORK_CROSS_ATTENTION_CACHE_GENERATOR_NAME]

        cross_attn_cache_generator = self.torch_class.TorchModule(model)

        torch.onnx.export(
            cross_attn_cache_generator,
            (encoder_hidden_states),
            output_fpath,
            do_constant_folding=True,
            opset_version=OPSET,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=TRAINING_MODE,
        )

        if network_metadata.precision.fp16:
            self.post_process_onnx(output_fpath)

        return self.onnx_class(model = output_fpath, network_metadata = network_metadata)

class Seq2SeqModelClass:
    """
    A class to track which class to use for each model type.
    """

    decoder_classes = {
        "torch": DecoderTorchFile,
        "onnx": DecoderONNXFile,
        "engine": DecoderTRTEngine
    }

    encoder_classes = {
        "torch": EncoderTorchFile,
        "onnx": EncoderONNXFile,
        "engine": EncoderTRTEngine
    }

    cross_attn_cache_generator_classes = {
        "torch": CrossAttnCacheGeneratorTorchFile,
        "onnx": CrossAttnCacheGeneratorONNXFile,
        "engine": CrossAttnCacheGeneratorTRTEngine
    }
