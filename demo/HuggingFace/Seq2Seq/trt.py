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

import os
import sys
import time

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# polygraphy
from polygraphy.backend.trt import Profile

# tensorrt
import tensorrt as trt

# torch
import torch

# huggingface
from transformers import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

# tensorrt
from tensorrt import PreviewFeature

# TRT-HuggingFace
from NNDF.interface import TRTInferenceCommand
from NNDF.networks import (
    NetworkMetadata,
    NetworkModels,
    NetworkModel,
    Precision,
)

from NNDF.general_utils import confirm_folder_delete
from NNDF.tensorrt_utils import TRTNativeRunner, setup_benchmark_arg
from NNDF.models import TRTEngineFile
from NNDF.logger import G_LOGGER

from Seq2Seq.Seq2SeqModelConfig import Seq2SeqModelTRTConfig
from Seq2Seq.measurements import calculate_perplexity_helper_encoder_decoder, calculate_perplexity_helper_decoder
from Seq2Seq.export import Seq2SeqModelClass

class Seq2SeqTRTEncoder(TRTNativeRunner):
    """TRT implemented network interface that can be used to measure inference time."""

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        config: Seq2SeqModelTRTConfig,
        nvtx_verbose: bool,
    ):
        super().__init__(trt_engine_file, network_metadata, config, nvtx_verbose)

        self.data_type = torch.float16 if network_metadata.precision.fp16 else torch.float32
        self.main_input_name = "input_ids"
        self.device = torch.device("cuda")

        self.bindings = [0] * self.trt_engine.num_bindings
        self.encoder_hidden_states = torch.zeros(
            config.batch_size*config.max_input_length*config.hidden_size,
            dtype=self.data_type,
            device=self.device
        )

        self.bindings[self.trt_engine.get_binding_index("encoder_hidden_states")] = self.encoder_hidden_states.data_ptr()

    def forward(self, input_ids: torch.Tensor):
        input_length = input_ids.shape[1]

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu"))

        if is_cpu_mode:
            input_ids = input_ids.int().flatten().contiguous().cuda()

        self.bindings[0] = input_ids.int().data_ptr()

        # Set the binding shape of input_ids, which should be (bs, input_length).
        self.trt_context.set_binding_shape(0, input_ids.shape)

        assert self.trt_context.all_binding_shapes_specified

        # Launch TRT inference.
        # TODO: Could we use execute_v2_async() instead of execute_v2()?
        self.trt_context.execute_v2(bindings=self.bindings)

        last_hidden_state = self.encoder_hidden_states[:self.config.batch_size * input_length * self.config.hidden_size].view(self.config.batch_size, input_length, self.config.hidden_size)

        if is_cpu_mode:
            last_hidden_state = last_hidden_state.cpu()

        return BaseModelOutput(last_hidden_state = last_hidden_state)

class Seq2SeqTRTDecoder(TRTNativeRunner, GenerationMixin):

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        network_metadata: NetworkMetadata,
        config: Seq2SeqModelTRTConfig,
        nvtx_verbose: bool,
    ):
        super().__init__(trt_engine_file, network_metadata, config, nvtx_verbose)

        self.main_input_name = "input_ids"
        self.generation_config = config.generation_config
        self.device = torch.device("cuda")
        self.encoder_hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.embedding_size_per_head = config.d_kv
        self.num_decoder_layers = config.num_decoder_layers
        self.expand_size = config.expand_size
        self.profile_idx = 0
        self.bindings = [0] * self.trt_engine.num_bindings

        # Construct buffer for logits outputs
        self.logits = torch.zeros(
            config.expand_size * config.max_decoder_length * config.vocab_size,
            dtype=config.precision,
            device=self.device,
        )

        self.bindings[self.trt_engine.get_binding_index("logits")] = self.logits.data_ptr()

        self.context_mode = (not config.is_encoder_decoder) and (config.use_cache)

        if config.is_encoder_decoder:
            self.encoder_hidden_states = torch.zeros(
                config.expand_size*config.max_input_length*config.hidden_size,
                dtype=config.precision,
                device=self.device
            )

            self.bindings[self.trt_engine.get_binding_index("encoder_hidden_states")] = self.encoder_hidden_states.data_ptr()

            self.encoder_length = 0
        else:
            self.num_bindings = self.trt_engine.num_bindings // 2 if self.config.use_cache else self.trt_engine.num_bindings

        if config.use_cache:

            self.self_attn_cache = {}
            self.past_cross_attn_cache = {}

            self_attn_cache_size = config.expand_size * config.num_heads * config.max_output_length * config.d_kv
            cross_attn_cache_size = config.expand_size * config.num_heads * config.max_input_length * config.d_kv

            # Set self attention kv cache shape and type
            for i in range(config.num_decoder_layers):
                for code in ["key", "value"]:
                    # Allocate self attention buffer. The buffer is used both as inputs and outputs
                    self_attn_name = f"key_values.{i}.self.{code}"
                    self_attn_buffer = [torch.zeros(
                        self_attn_cache_size,
                        dtype=config.precision,
                        device=self.device,
                    ), torch.zeros(
                        self_attn_cache_size,
                        dtype=config.precision,
                        device=self.device,
                    )]
                    # Set input = output for cache. Might break for GPT/BART. Let's give it a try first.
                    input_idx = self.trt_engine.get_binding_index("past_" + self_attn_name)
                    self.self_attn_cache[self_attn_name] = self_attn_buffer
                    self.bindings[input_idx] = self_attn_buffer[0].data_ptr()

                    output_idx = self.trt_engine.get_binding_index("present_" + self_attn_name)
                    self.bindings[output_idx] = self_attn_buffer[1].data_ptr()

                    if config.is_encoder_decoder:
                        # Allocate cross attention buffer
                        cross_attn_past_name = f"past_key_values.{i}.cross.{code}"
                        cross_attn_buffer = torch.zeros(
                            cross_attn_cache_size,
                            dtype=config.precision,
                            device=self.device,
                        )
                        cross_attn_idx = self.trt_engine.get_binding_index(cross_attn_past_name)
                        self.past_cross_attn_cache[cross_attn_past_name] = cross_attn_buffer
                        self.bindings[cross_attn_idx] = cross_attn_buffer.data_ptr()
                    else:
                        # Context mode will always use buffer same buffer as output
                        self.bindings[input_idx + self.num_bindings] = 0 # Context phase, should be 0
                        self.bindings[output_idx + self.num_bindings] = self_attn_buffer[0].data_ptr()

                    self.cache_id = 0

            self.cache_binding_offset = 2 if config.is_encoder_decoder else 1
            if not self.config.is_encoder_decoder:
                self.set_context_mode_trt_context()
                self.bindings[self.trt_engine.get_binding_index("logits") + self.num_bindings] = self.logits.data_ptr()

            self.num_cache_per_layer = 4 if config.is_encoder_decoder else 2
            self.past_decoder_length = 0

    def can_generate(self):
        return True

    def set_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor):
        """Used to cache encoder hidden state runs across same encoder sessions"""
        self.encoder_hidden_states[:encoder_hidden_states.numel()] = encoder_hidden_states.flatten().contiguous()
        self.trt_context.set_binding_shape(1, encoder_hidden_states.shape)

    def set_cross_attn_cache_generator_engine(self, cross_attn_cache_generator):

        # Sharing memory between 2 TRT context may have memory corruption. Therefore create separate memory for cross attn generator
        self.cross_attn_cache_generator = cross_attn_cache_generator
        with open(self.cross_attn_cache_generator.fpath, "rb") as f:
            trt_runtime = trt.Runtime(self.trt_logger)
            self.cross_attn_cache_generator_engine = trt_runtime.deserialize_cuda_engine(f.read())
            self.cross_attn_cache_generator_context = self.cross_attn_cache_generator_engine.create_execution_context()
            self.cross_attn_cache_generator_context.nvtx_verbosity = self.trt_context.nvtx_verbosity

        self.cross_attn_bindings = [0] * self.cross_attn_cache_generator_engine.num_bindings
        self.cross_attn_bindings[0] = self.encoder_hidden_states.data_ptr()
        # Cross attention cache as outputs
        for i in range(self.num_decoder_layers):
            self.cross_attn_bindings[2*i+1] = self.past_cross_attn_cache[f"past_key_values.{i}.cross.key"].data_ptr()
            self.cross_attn_bindings[2*i+2] = self.past_cross_attn_cache[f"past_key_values.{i}.cross.value"].data_ptr()

    def set_cross_attn_cache(self, encoder_hidden_states: torch.Tensor):
        """
        Used to cache encoder-decoder cross attention kv caches across same encoder sessions.

        Unlike self-attention cache, cross attention is constant during the decoding process, so we only need to set its bindings once at the first decoding step, and skip in all later steps (by self.persist_cross_attention_kv_cache flag)
        """

        self.cross_attn_cache_generator_context.set_binding_shape(0, encoder_hidden_states.shape)

        assert self.cross_attn_cache_generator_context.all_binding_shapes_specified
        self.cross_attn_cache_generator_context.execute_v2(bindings=self.cross_attn_bindings)

        encoder_length = encoder_hidden_states.shape[1]
        cross_attn_cache_shape = (self.expand_size, self.num_heads, encoder_length, self.embedding_size_per_head)

        for i in range(self.num_decoder_layers):

            self.trt_context.set_binding_shape(self.cache_binding_offset+4*i+2, cross_attn_cache_shape)
            self.trt_context.set_binding_shape(self.cache_binding_offset+4*i+3, cross_attn_cache_shape)

    def set_context_mode_trt_context(self):
        # Create TRT context for context mode (1st decoder run) with optimization profile = 1
        self.context_trt_context = self.trt_engine.create_execution_context()
        self.context_trt_context.active_optimization_profile = 1

    def forward(
        self,
        input_ids,
        encoder_outputs: BaseModelOutput = None,
        **kwargs
    ) -> Seq2SeqLMOutput:

        # Actual sequence length of the input_ids and the output hidden_states.
        input_length = input_ids.shape[1]

        if self.config.is_encoder_decoder:
            if encoder_outputs is None:
                raise RuntimeError("You are using a encoder_decoder model but does not provice encoder_outputs.")
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_length = encoder_hidden_states.shape[1]
        else:
            # When seq len != 1 for decoder only model, meaning it is context mode.
            if self.config.use_cache and input_length > 1:
                self.context_mode = True

        input_shape = input_ids.shape
        is_cpu_mode = input_ids.device == torch.device("cpu")

        input_ids = input_ids.int()

        if is_cpu_mode:
            input_ids = input_ids.cuda()

        if not self.context_mode:
            self.bindings[0] = input_ids.data_ptr()
            self.trt_context.set_binding_shape(0, input_shape)
        else:
            self.bindings[self.num_bindings] = input_ids.data_ptr()
            self.context_trt_context.set_binding_shape(self.num_bindings, input_shape)

        if self.config.use_cache:
            if kwargs.get("past_key_values") is None:
                # If no past_key_values are passed in from prepare_inputs_for_generation, need to reset encoder_hidden_states and cross_attn_cache
                self.past_decoder_length = 0
                if self.config.is_encoder_decoder:
                    self.set_encoder_hidden_states(encoder_hidden_states)
                    self.set_cross_attn_cache(encoder_hidden_states)

            self_attn_cache_shape = (self.expand_size, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)

            for i in range(self.num_decoder_layers):
                if self.context_mode:
                    self.context_trt_context.set_binding_shape(self.cache_binding_offset+2*i+self.num_bindings, self_attn_cache_shape)
                    self.context_trt_context.set_binding_shape(self.cache_binding_offset+2*i+1+self.num_bindings, self_attn_cache_shape)
                else:
                    self.trt_context.set_binding_shape(self.cache_binding_offset+self.num_cache_per_layer*i, self_attn_cache_shape)
                    self.trt_context.set_binding_shape(self.cache_binding_offset+self.num_cache_per_layer*i+1, self_attn_cache_shape)

        elif input_length == 1 and self.config.is_encoder_decoder:
            encoder_hidden_states = encoder_outputs.last_hidden_state
            self.set_encoder_hidden_states(encoder_hidden_states)

        # Launch TRT inference.
        if self.context_mode:
            assert self.context_trt_context.all_binding_shapes_specified
            self.context_trt_context.execute_v2(bindings=self.bindings)
        else:
            assert self.trt_context.all_binding_shapes_specified
            self.trt_context.execute_v2(bindings=self.bindings)

        # For bs > 1, this is required, so cannnot avoid this D2D copy
        logits_length = self.expand_size * input_length * self.config.vocab_size
        logits = self.logits[:logits_length].view(self.expand_size, input_length, self.config.vocab_size)

        present_key_values = None
        if self.config.use_cache:

            self._switch_cache()

            present_key_values = ()
            self.past_decoder_length += input_length

            if self.context_mode:
                result_id = 0
            else:
                result_id = self.cache_id

            self_attn_cache_shape = (self.expand_size, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)
            self_attn_cache_size = self.expand_size * self.num_heads * self.past_decoder_length * self.embedding_size_per_head

            for i in range(self.num_decoder_layers):
                self_attn_k_output = self.self_attn_cache[f"key_values.{i}.self.key"][result_id]
                self_attn_v_output = self.self_attn_cache[f"key_values.{i}.self.value"][result_id]

                present_key_values += ((self_attn_k_output, self_attn_v_output),)

            if self.context_mode:
                self.context_mode = False

        return Seq2SeqLMOutput(logits=logits, past_key_values=present_key_values,)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, encoder_outputs = None, **kwargs):
        # In HuggingFace generation_utils.py, this function will be called at each decoding step, before running the decoder's forward().
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        input_args = {
            "input_ids": input_ids,
        }

        if self.config.is_encoder_decoder:
            input_args["encoder_outputs"] = encoder_outputs

        if self.config.use_cache:
            input_args["use_cache"] = True
            input_args["past_key_values"] = past_key_values

        return input_args

    def _switch_cache(self):

        if not (self.cache_id == 0 and self.context_mode):
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:
                    self_attention_name = f"key_values.{i}.self.{code}"
                    input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                    output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)

                    temp = self.bindings[output_idx]
                    self.bindings[output_idx] = self.bindings[input_idx]
                    self.bindings[input_idx] = temp

            self.cache_id = (self.cache_id + 1) % 2

    def _reorder_cache(self, past, beam_idx):

        # `past` does not change, but we have reordered the cache within the class.
        if past is None:
            G_LOGGER.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        self_attn_output_shape = (self.expand_size, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)
        self_attn_output_size = self.expand_size * self.num_heads * (self.past_decoder_length) * self.embedding_size_per_head


        # TODO: This step has significant D2D memory copy.
        for i in range(self.num_decoder_layers):

            k = self.self_attn_cache[f"key_values.{i}.self.key"][self.cache_id][:self_attn_output_size].view(*self_attn_output_shape)
            v = self.self_attn_cache[f"key_values.{i}.self.value"][self.cache_id][:self_attn_output_size].view(*self_attn_output_shape)

            reordered_k = k.index_select(0, beam_idx.to(k.device))
            reordered_v = v.index_select(0, beam_idx.to(v.device))

            self.self_attn_cache[f"key_values.{i}.self.key"][self.cache_id][:self_attn_output_size] = reordered_k.flatten().contiguous()
            self.self_attn_cache[f"key_values.{i}.self.value"][self.cache_id][:self_attn_output_size] = reordered_v.flatten().contiguous()

        return past

class Seq2SeqTRT(TRTInferenceCommand):
    def __init__(
        self,
        config_class=Seq2SeqModelTRTConfig,
        description="Runs trt results for Seq2Seq model.",
        model_classes=Seq2SeqModelClass,
        **kwargs
    ):
        super().__init__(
            network_config=config_class,
            description=description,
            model_classes=model_classes,
            **kwargs
        )
        self.onnx_encoder = None
        self.onnx_decoder = None

    def process_framework_specific_arguments(
        self,
        disable_preview_dynamic_shapes: bool = False,
        encoder_engine: str = None,
        decoder_engine: str = None,
        cache_generator_engine: str = None,
        use_timing_cache: bool = False,
        nvtx_verbose: bool = False,
        **kwargs
    ):

        self.disable_preview_dynamic_shapes = disable_preview_dynamic_shapes
        self.use_generator = self.config.is_encoder_decoder and self.config.use_cache

        self.workspace.set_encoder_engine_path(encoder_engine)
        self.workspace.set_decoder_engine_path(decoder_engine)
        self.workspace.set_cross_attn_generator_engine_path(cache_generator_engine)

        self.use_timing_cache = use_timing_cache
        self.timing_cache = self.workspace.get_timing_cache() if self.use_timing_cache else None

        self.nvtx_verbose_build = True # In building the engine, setting nvtx verbose level does not affect performance, so always set to True.
        self.nvtx_verbose_inference = nvtx_verbose # In inference, nvtx verbose level may affect performance.

        if self.use_generator:
            self.cross_attn_cache_generator_engine = None
            self.onnx_cross_attn_cache_generator = None
        return kwargs


    def setup_tokenizer_and_model(self):
        """
        Set up tokenizer and TRT engines for TRT Runner.
        """
        self.tokenizer = self.download_tokenizer()
        t0 = time.time()
        # Check whether user passed engine
        if self.check_engine_inputs_valid() and self.setup_engines_from_path(
            encoder_engine_fpath=self.workspace.encoder_engine_path,
            decoder_engine_fpath=self.workspace.decoder_engine_path,
            cross_attn_cache_generator_engine_fpath=self.workspace.cross_attn_generator_engine_path
        ):
            G_LOGGER.info("TRT engine loaded successful from arguments. Engine loading time: {:.4f}s".format(time.time() - t0))
        else:
            G_LOGGER.info("Cannot load existing TRT engines from arguments. Attempt to obtain from onnx model.")
            self.workspace.create_onnx_folders()
            self.load_onnx_model()
            self.setup_engines_from_onnx()
            G_LOGGER.info("TRT engine successfully obtained from onnx models. Total engine loading/building time: {:.4f}s".format(time.time() - t0))

        trt_models = [
            NetworkModel(
                name=self.config.NETWORK_DECODER_SEGMENT_NAME,
                fpath=self.decoder_engine.fpath,
            )
        ]

        if self.config.is_encoder_decoder:
            trt_models.append(
                NetworkModel(
                    name=self.config.NETWORK_ENCODER_SEGMENT_NAME,
                    fpath=self.encoder_engine.fpath,
                )
            )

        if self.use_generator:
            trt_models.append(
                NetworkModel(
                    name=self.config.NETWORK_CROSS_ATTENTION_CACHE_GENERATOR_NAME,
                    fpath=self.cross_attn_cache_generator_engine.fpath,
                )
            )

        return NetworkModels(torch=None, onnx=None, trt=trt_models)

    def check_engine_inputs_valid(self):
        """
        Check whether all engines are valid.
        """
        encoder_engine_fpath = self.workspace.encoder_engine_path
        decoder_engine_fpath = self.workspace.decoder_engine_path
        cache_generator_engine_fpath = self.workspace.cross_attn_generator_engine_path
        is_encoder_valid = encoder_engine_fpath is not None and os.path.exists(encoder_engine_fpath)
        is_decoder_valid = decoder_engine_fpath is not None and os.path.exists(decoder_engine_fpath)
        is_generator_valid = cache_generator_engine_fpath is not None and os.path.exists(cache_generator_engine_fpath)
        if self.config.is_encoder_decoder:
            if self.config.use_cache:
                return is_encoder_valid and is_decoder_valid and is_generator_valid
            else:
                return is_encoder_valid and is_decoder_valid

        return is_decoder_valid

    def setup_engines_from_path(
        self,
        encoder_engine_fpath = None,
        decoder_engine_fpath = None,
        cross_attn_cache_generator_engine_fpath = None
    ):
        """
        Check whether user has passed in all required TRT engine name.
        If user passed valid TRT engines, will skip onnx export and use engine directly.
        """
        if decoder_engine_fpath is None:
            return False
        if self.config.is_encoder_decoder and encoder_engine_fpath is None:
            return False
        if self.use_generator and cross_attn_cache_generator_engine_fpath is None:
            return False

        try:
            self.decoder_engine = self.config.decoder_classes["engine"](decoder_engine_fpath, self.metadata)
            self.decoder = Seq2SeqTRTDecoder(
                self.decoder_engine, self.metadata, self.config, self.nvtx_verbose_inference
            )

            if self.config.is_encoder_decoder:
                self.encoder_engine = self.config.encoder_classes["engine"](encoder_engine_fpath, self.metadata)
                if self.config.use_fp32_encoder:
                    encoder_metadata = self.metadata._replace(precision=Precision(fp16=False))
                else:
                    encoder_metadata = self.metadata

                self.encoder = Seq2SeqTRTEncoder(
                    self.encoder_engine, encoder_metadata, self.config, self.nvtx_verbose_inference
                )

            if self.use_generator:
                self.cross_attn_cache_generator_engine = self.config.cross_attn_cache_generator_classes["engine"](cross_attn_cache_generator_engine_fpath, self.metadata)
                self.decoder.set_cross_attn_cache_generator_engine(self.cross_attn_cache_generator_engine)

            return True
        except Exception as e:
            G_LOGGER.error("Cannot proceed with the provided engine. Attempt to generate from onnx. Reason is: {}".format(str(e)))

        return False

    def setup_engines_from_onnx(self) -> None:
        """
        Set up TRT engines from onnx files.
        """

        # Generate optimization profiles.
        # non-benchmarking mode: opt profile length is by default half of the max profile
        # benchmarking mode: user can specify opt and max profile by flags. If no additional benchmarking flags are provided, it will just use the non-benchmarking mode defaults
        max_input_length = self.config.max_length
        max_output_length = self.config.max_length
        opt_input_seq_len = max_input_length // 2
        opt_output_seq_len = max_output_length // 2

        # benchmarking flags
        if self.benchmarking_mode:
            opt_input_seq_len, opt_output_seq_len, max_input_length, max_output_length, seq_tag = self.process_benchmarking_args()

        encoder_hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        expand_size = self.config.expand_size
        is_encoder_decoder = self.config.is_encoder_decoder
        use_cache = self.config.use_cache
        num_beams = self.config.num_beams

        # Convert ONNX models to TRT engines.
        if not self.benchmarking_mode:
            engine_tag = "bs{}".format(batch_size)
        # When user does not input any profile_max_len, use seq as tag, both max are config max
        elif seq_tag:
            engine_tag = "bs{}-inseq{}-outseq{}".format(batch_size, opt_input_seq_len, opt_output_seq_len)
        # When user input profile_max_len, reuse the engine for future use with different seq_len
        else:
            engine_tag = "bs{}-inmax{}-outmax{}".format(batch_size, max_input_length, max_output_length)

        if num_beams > 1:
            engine_tag += "-beam{}".format(num_beams)

        preview_features = [PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]

        if self.disable_preview_dynamic_shapes:
            engine_tag += "-noPreviewFasterDynamicShapes"
        else:
            preview_features.append(PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)

        # Set up decoder engine
        if not use_cache:
            decoder_profile = Profile().add(
                "input_ids",
                min=(expand_size, 1),
                opt=(expand_size, opt_output_seq_len),
                max=(expand_size, max_output_length),
            )
            if is_encoder_decoder:
                decoder_profile.add(
                    "encoder_hidden_states",
                    min=(expand_size, 1, encoder_hidden_size),
                    opt=(expand_size, opt_input_seq_len, encoder_hidden_size),
                    max=(expand_size, max_input_length, encoder_hidden_size),
                )

            decoder_profiles = [decoder_profile]
        else:
            num_heads = self.config.num_heads
            embedding_size_per_head = self.config.d_kv
            num_decoder_layers = self.config.num_decoder_layers

            self_attn_profile = {
                "min": (expand_size, num_heads, 0, embedding_size_per_head),
                "opt": (expand_size, num_heads, opt_output_seq_len - 1, embedding_size_per_head),
                "max": (expand_size, num_heads, max_output_length - 1, embedding_size_per_head),
            }

            cross_attn_profile = {
                "min": (expand_size, num_heads, 1, embedding_size_per_head),
                "opt": (expand_size, num_heads, opt_input_seq_len, embedding_size_per_head),
                "max": (expand_size, num_heads, max_input_length, embedding_size_per_head),
            }

            decoder_profile_generation = Profile().add(
                "input_ids",
                min=(expand_size, 1),
                opt=(expand_size, 1),
                max=(expand_size, 1),
            )

            if is_encoder_decoder:
                decoder_profile_generation.add(
                    "encoder_hidden_states",
                    min=(expand_size, 1, encoder_hidden_size),
                    opt=(expand_size, opt_input_seq_len, encoder_hidden_size),
                    max=(expand_size, max_input_length, encoder_hidden_size),
                )


            for i in range(num_decoder_layers):
                decoder_profile_generation = decoder_profile_generation.add(
                    f"past_key_values.{i}.self.key",
                    **self_attn_profile
                ).add(
                    f"past_key_values.{i}.self.value",
                    **self_attn_profile
                )
                if is_encoder_decoder:
                    decoder_profile_generation = decoder_profile_generation.add(
                        f"past_key_values.{i}.cross.key",
                        **cross_attn_profile
                    ).add(
                        f"past_key_values.{i}.cross.value",
                        **cross_attn_profile
                    )

            decoder_profiles = [decoder_profile_generation]

            # Decoder only model has "context phase" that is only used for the 1st decoder phase.
            if not is_encoder_decoder:
                # Context phase takes various-length input_ids with no kv cache and generates initial cache for subsequent decoding steps.
                decoder_profile_context = Profile().add(
                    "input_ids",
                    min=(expand_size, 1),
                    opt=(expand_size, opt_input_seq_len),
                    max=(expand_size, max_input_length),
                )

                self_attn_profile_context = {
                    "min": (expand_size, num_heads, 0, embedding_size_per_head),
                    "opt": (expand_size, num_heads, 0, embedding_size_per_head),
                    "max": (expand_size, num_heads, 0, embedding_size_per_head),
                }
                for i in range(num_decoder_layers):
                    decoder_profile_context = decoder_profile_context.add(
                        f"past_key_values.{i}.self.key",
                        **self_attn_profile_context
                    ).add(
                        f"past_key_values.{i}.self.value",
                        **self_attn_profile_context
                    )

                decoder_profiles.append(decoder_profile_context)

        decoder_engine_path = self.workspace.get_engine_fpath_from_onnx(self.onnx_decoder.fpath, engine_tag)

        G_LOGGER.info("Setting up decoder engine in {}...".format(decoder_engine_path))

        self.decoder_engine = self.onnx_decoder.as_trt_engine(
            decoder_engine_path,
            profiles = decoder_profiles,
            preview_features = preview_features,
            nvtx_verbose = self.nvtx_verbose_build,
            timing_cache = self.timing_cache,
        )

        self.decoder = Seq2SeqTRTDecoder(
            self.decoder_engine, self.metadata, self.config, self.nvtx_verbose_inference,
        )

        self.workspace.set_decoder_engine_path(decoder_engine_path)
        # Set up encoder if needed
        if is_encoder_decoder:
            encoder_profiles = [
                Profile().add(
                    "input_ids",
                    min=(batch_size, 1),
                    opt=(batch_size, opt_input_seq_len),
                    max=(batch_size, max_input_length),
                )
            ]
            encoder_engine_path = self.workspace.get_engine_fpath_from_onnx(self.onnx_encoder.fpath, engine_tag).replace(f"-beam{num_beams}", "")

            G_LOGGER.info("Setting up encoder engine in {}...".format(encoder_engine_path))

            self.encoder_engine = self.onnx_encoder.as_trt_engine(
                encoder_engine_path, # encoder engine name not affected by beam search
                profiles=encoder_profiles,
                preview_features=preview_features,
                nvtx_verbose=self.nvtx_verbose_build,
                timing_cache=self.timing_cache,
            )

            if self.config.use_fp32_encoder:
                encoder_metadata = self.metadata._replace(precision=Precision(fp16=False))
            else:
                encoder_metadata = self.metadata

            self.encoder = Seq2SeqTRTEncoder(
                self.encoder_engine, encoder_metadata, self.config, self.nvtx_verbose_inference,
            )
            self.workspace.set_encoder_engine_path(encoder_engine_path)

        # Set up cross attn cache generator if needed
        if self.use_generator:
            generator_profiles = [Profile().add(
                "encoder_hidden_states",
                min=(expand_size, 1, encoder_hidden_size),
                opt=(expand_size, opt_input_seq_len, encoder_hidden_size),
                max=(expand_size, max_input_length, encoder_hidden_size),
            )]

            cross_attn_cache_generator_engine_path = self.workspace.get_engine_fpath_from_onnx(self.onnx_cross_attn_cache_generator.fpath, engine_tag)

            G_LOGGER.info("use_cache=True. Setting up cross attention kv cache generator in {}...".format(cross_attn_cache_generator_engine_path))
            self.cross_attn_cache_generator_engine = self.onnx_cross_attn_cache_generator.as_trt_engine(
                cross_attn_cache_generator_engine_path,
                profiles=generator_profiles,
                preview_features=preview_features,
                nvtx_verbose=self.nvtx_verbose_build,
                timing_cache=self.timing_cache,
            )
            self.decoder.set_cross_attn_cache_generator_engine(self.cross_attn_cache_generator_engine)
            self.workspace.set_cross_attn_generator_engine_path(cross_attn_cache_generator_engine_path)

    def process_benchmarking_args(self):
        # For benchmarking mode, need to set max length large.
        max_input_seq_len = self.config.n_positions
        max_output_seq_len = self.config.n_positions
        seq_tag = self._args.input_profile_max_len is None and self._args.output_profile_max_len is None
        # User must provide either a pair of profile_max_len or a profile of seq_len for input/output
        if self._args.input_profile_max_len is None or self._args.output_profile_max_len is None:
            if self._args.input_seq_len is None or self._args.output_seq_len is None:
                raise RuntimeError("Please provide at least one pair of inputs: [input/output]_seq_len or [input/output]_profile_max_len")

        input_profile_max_len = setup_benchmark_arg(self._args.input_profile_max_len, "input_profile_max_len", max_input_seq_len)
        output_profile_max_len = setup_benchmark_arg(self._args.output_profile_max_len, "output_profile_max_len", max_output_seq_len)
        input_seq_len = setup_benchmark_arg(self._args.input_seq_len, "input_seq_len", input_profile_max_len // 2)
        output_seq_len = setup_benchmark_arg(self._args.output_seq_len, "output_seq_len", output_profile_max_len // 2)

        # Assert to ensure the validity of benchmarking arguments
        assert input_seq_len <= input_profile_max_len, "input_seq_len should <= input_profile_max_len = {} for benchmarking mode".format(input_profile_max_len)
        assert output_seq_len <= output_profile_max_len, "output_seq_len should <= output_profile_max_len = {} for benchmarking mode".format(output_profile_max_len)
        assert input_profile_max_len <= max_input_seq_len, "Model config restrict input_profile_max_len <= {} for benchmark mode".format(max_input_seq_len)
        assert output_profile_max_len <= max_output_seq_len, "Model config restrict output_profile_max_len <= {} for benchmark mode".format(max_output_seq_len)

        return input_seq_len, output_seq_len, input_profile_max_len, output_profile_max_len, seq_tag

    def calculate_perplexity(self, input_str: str, reference_str: str, use_cuda: bool = True):

        if self.config.use_cache or self.config.num_beams > 1:
            G_LOGGER.warning("Perplexity calculation is disabled for use_cache=True or num_beams>1 in TRT. Default=None")
            return None

        if self.config.is_encoder_decoder:
            perplexity = calculate_perplexity_helper_encoder_decoder(
                encoder=self.encoder,
                decoder=self.decoder,
                tokenizer=self.tokenizer,
                input_str=input_str,
                reference_str=reference_str,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                use_cuda=use_cuda,
            )
        else:
            perplexity = calculate_perplexity_helper_decoder(
                decoder=self.decoder,
                tokenizer=self.tokenizer,
                input_str=reference_str,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                use_cuda=use_cuda,
            )

            G_LOGGER.info("Perplexity={}".format(perplexity))

        return perplexity

    def cleanup(self) -> None:

        # Deactivates context
        if self.encoder:
            self.encoder.release()
        if self.decoder:
            self.decoder.release()

        if not self.keep_trt_engine:
            self.decoder_engine.cleanup()
            if self.config.is_encoder_decoder:
                self.encoder_engine.cleanup()
            if self.use_generator:
                self.cross_attn_cache_generator_engine.cleanup()

        if not self.keep_onnx_model:
            if self.onnx_decoder:
                self.onnx_decoder.cleanup()
            if self.config.is_encoder_decoder and self.onnx_encoder:
                self.onnx_encoder.cleanup()
            if self.use_generator and self.onnx_cross_attn_cache_generator:
                self.onnx_cross_attn_cache_generator.cleanup()

        if not self.keep_torch_model and self.workspace.torch_path is not None:

            confirm_folder_delete(
                self.workspace.torch_path,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

        if not self.keep_trt_engine:
            self.workspace.cleanup(force_remove=False)


RUN_CMD = Seq2SeqTRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))