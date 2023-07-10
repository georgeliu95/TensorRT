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
import sys

import numpy as np
import tensorrt as trt
import torch
from transformers.configuration_utils import PretrainedConfig

sys.path.append('../../HuggingFace') # Include HuggingFace directory
from NNDF.models import TRTEngineFile
from NNDF.networks import NetworkMetadata
from NNDF.tensorrt_utils import TRTNativeRunner
from NNDF.logger import G_LOGGER
from Seq2Seq.export import DecoderTRTEngine

from HuggingFace.NNDF.tensorrt_utils import TRTNativeRunner, CUASSERT
from cuda import cudart


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
        self.input_tensor_names = set()
        for i in range(self.trt_engine.num_io_tensors):
            tensor_name = self.trt_engine.get_tensor_name(i)
            self.io_names.add(tensor_name)
            if self.trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_tensor_names.add(tensor_name)

        self.cfg = cfg
        logits_size = self.cfg.batch_size * self.cfg.model.max_seq_len * self.cfg.model.vocab_size
        dtype = self.get_torch_type(self.get_output_name())
        self.logits = torch.zeros(logits_size, dtype=dtype).contiguous().cuda()

    def _set_context_mode_trt_context(self):
        # Create TRT context for context mode (1st decoder run) with optimization profile index = 1
        self.context_trt_context = self.trt_engine.create_execution_context()
        self.context_trt_context.active_optimization_profile = 1
        self.kv_cache_binding_offset = self.trt_engine.num_bindings // self.trt_engine.num_optimization_profiles

    def get_torch_type(self, name):
        trt_type = self.trt_engine.get_binding_dtype(name)
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
            trt.uint8: torch.uint8,
            trt.bfloat16: torch.bfloat16,
        }
        if trt_type in mapping:
            return mapping[trt_type]
        raise ValueError(f"Got unexpected tensorrt dtype {trt_type} in get_torch_type().")

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
        if self.ATTENTION_MASK_INDEX < self.trt_engine.num_bindings:
            return self.trt_engine.get_binding_name(self.ATTENTION_MASK_INDEX) == "attention_mask"
        return False

    def get_attention_mask_name(self):
        if self.has_attention_mask():
            return self.trt_engine.get_binding_name(self.ATTENTION_MASK_INDEX)
        return None

    def run(self, output_name, io_descs, is_first_input=False):
        # Set active optimization profile and active execution context.
        self.trt_context.active_optimization_profile = self.profile_idx
        active_context = self.trt_context
        if is_first_input and self.use_cache:
            active_context = self.context_trt_context

        # Set up input bindings.
        for name, tensor_shape in io_descs.items():
            active_context.set_tensor_address(name, tensor_shape[0])
            if name in self.input_tensor_names:
                active_context.set_input_shape(name, tensor_shape[1])
            elif self.use_cache:
                pass
            else:
                assert False, "All tensors must be inputs for non-KV mode"
        assert active_context.all_shape_inputs_specified

        # Set up output bindings.
        assert output_name == self.get_output_name()
        engine_out_torch_type = self.get_torch_type(output_name)
        if self.logits.dtype != engine_out_torch_type:
            raise ValueError(f"Output data type does not match, {self.logits.dtype} vs. {engine_out_torch_type}.")
        shape = active_context.get_tensor_shape(output_name)
        active_context.set_tensor_address(output_name, self.logits.data_ptr())


        # Execute inference.
        active_context.execute_async_v3(self.stream)
        CUASSERT(cudart.cudaStreamSynchronize(self.stream))
        if len(shape) != 3:
            raise ValueError("Output must have a dimension of 3.")
        output = self.logits[:shape[0] * shape[1] * shape[2]].view(tuple(shape))
        return output

def load_trt_model(cfg):
    G_LOGGER.info(f'Loading TensorRT engine from {cfg.trt_engine_file} with use_cache={cfg.use_cache}')
    trt_engine_file = DecoderTRTEngine(cfg.trt_engine_file)
    return GPTTRTDecoder(trt_engine_file, cfg.use_cache, cfg)