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
import torch
import numpy as np
import tensorrt as trt

import sys
sys.path.append('../') # Include one-level up directory so to reuse HuggingFace utils.
sys.path.append('../HuggingFace') # Include HuggingFace directory as well
from HuggingFace.NNDF.models import TRTEngineFile
from HuggingFace.Seq2Seq.export import DecoderTRTEngine
from HuggingFace.NNDF.networks import NetworkMetadata
from HuggingFace.NNDF.tensorrt_utils import TRTNativeRunner

from transformers.configuration_utils import PretrainedConfig

def to_torch_dtype(np_type):
    if np_type == np.int32:
        return torch.int32
    elif np_type == np.int64:
        return torch.int64
    elif np_type == np.float16:
        return torch.float16
    elif np_type == np.float or np_type == np.float32:
        return torch.float32
    elif np_type == np.bool or np_type == np.bool_:
        return torch.bool
    else:
        raise ValueError(f"Got unexpected numpy dtype {np_type} in to_torch_dtype().")

class GPTTRTDecoder(TRTNativeRunner):

    INPUT_IDS_INDEX = 0
    POSITION_IDS_INDEX = 1
    ATTENTION_MASK_INDEX = 2

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        use_cache: bool,
        cfg,
        network_metadata: NetworkMetadata = None,
        hf_config: PretrainedConfig = None,
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config)
        self.use_cache = use_cache
        if self.use_cache:
            self._set_context_mode_trt_context()
        self.io_names = set()
        for i in range(self.trt_engine.num_bindings):
            self.io_names.add(self.trt_engine.get_binding_name(i))

        self.cfg = cfg
        logits_size = self.cfg.batch_size * self.cfg.model.max_seq_len * self.cfg.model.vocab_size
        self.logits = torch.zeros(logits_size, dtype=torch.float16).contiguous().cuda()

    def _set_context_mode_trt_context(self):
        # Create TRT context for context mode (1st decoder run) with optimization profile index = 1
        self.context_trt_context = self.trt_engine.create_execution_context()
        self.context_trt_context.active_optimization_profile = 1
        self.kv_cache_binding_offset = self.trt_engine.num_bindings // self.trt_engine.num_optimization_profiles

    def get_type(self, name):
        return trt.nptype(self.trt_engine.get_binding_dtype(name))

    def get_input_ids_name(self):
        return self.trt_engine.get_binding_name(self.INPUT_IDS_INDEX)

    def has_position_ids(self):
        # If the input at POSITION_IDS_INDEX has a dimension of 2, assume it is position_ids.
        return len(self.trt_engine.get_binding_shape(self.POSITION_IDS_INDEX)) == 2

    def get_position_ids_name(self):
        if self.has_position_ids():
            return self.trt_engine.get_binding_name(self.POSITION_IDS_INDEX)
        else:
            return None

    def get_output_name(self):
        if self.use_cache:
            return self.trt_engine.get_binding_name(self.kv_cache_binding_offset - 1)
        return self.trt_engine.get_binding_name(self.trt_engine.num_bindings - 1)

    def has_attention_mask(self):
        # If the input at ATTENTION_MASK_INDEX has a dimension of 2, assume it is attention_mask.
        return len(self.trt_engine.get_binding_shape(self.ATTENTION_MASK_INDEX)) == 2

    def get_attention_mask_name(self):
        if self.has_attention_mask():
            return self.trt_engine.get_binding_name(self.ATTENTION_MASK_INDEX)
        else:
            return None

    def run(self, output_names, io_descs, is_first_input=False):
        def get_binding_idx(name, binding_offset=0):
            idx = self.trt_engine.get_binding_index(name)
            return binding_offset + idx

        # Set active optimization profile and active execution context.
        self.trt_context.active_optimization_profile = self.profile_idx
        active_context = self.trt_context
        binding_offset = 0
        if is_first_input and self.use_cache:
            active_context = self.context_trt_context
            binding_offset += self.kv_cache_binding_offset

        # Set up input bindings.
        bindings = [0] * self.trt_engine.num_bindings
        tensors = []
        for name, tensor_shape in io_descs.items():
            tensors.append(tensor_shape[0])

            idx = get_binding_idx(name, binding_offset)
            bindings[idx] = tensor_shape[0].data_ptr()
            if self.trt_engine.binding_is_input(name):
                active_context.set_binding_shape(idx, tensor_shape[1])
            elif self.use_cache:
                pass
            else:
                assert False, "All tensors must be inputs for non-KV mode"
        assert active_context.all_binding_shapes_specified

        # Set up output bindings.
        assert len(output_names) == 1 and output_names[0] == self.get_output_name()
        type = trt.nptype(self.trt_engine.get_binding_dtype(name))
        if self.logits.dtype != to_torch_dtype(type):
            raise ValueError(f"Output data type does not match, {self.logits.dtype} vs. {to_torch_dtype(type)}.")
        idx = get_binding_idx(output_names[0], binding_offset)
        shape = active_context.get_binding_shape(idx)
        bindings[idx] = self.logits.data_ptr()

        # Execute inference.
        active_context.execute_v2(bindings=bindings)
        if len(shape) != 3:
            raise ValueError("Output must have a dimension of 3.")
        output = self.logits[:shape[0] * shape[1] * shape[2]].view(tuple(shape))
        return [output]

def load_trt_model(cfg):
    print(f'loading trt model {cfg.trt_engine_file} enable_kv_cache {cfg.enable_kv_cache}')
    trt_engine_file = DecoderTRTEngine(cfg.trt_engine_file)
    return GPTTRTDecoder(trt_engine_file, cfg.enable_kv_cache, cfg)
