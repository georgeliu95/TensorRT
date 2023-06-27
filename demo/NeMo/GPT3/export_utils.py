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
import warnings
from typing import Dict, List, Optional
import numpy as np

# nemo
from nemo.core import ModelPT
from nemo.core.config import hydra_runner
from nemo.core.classes import Exportable
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.export_utils import augment_filename
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel, MegatronGPTExportableModel

# onnx
import onnx
from onnx import helper, TensorProto, numpy_helper, version_converter
import onnx_graphsurgeon as gs
    
# polygraphy
from polygraphy.backend.trt import Profile, CreateConfig, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.logger import G_LOGGER as PG_LOGGER

# tensorrt
from tensorrt import PreviewFeature
import torch
import transformer_engine

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from GPT3.nemo_utils import load_nemo_model, release_nemo_model

# HuggingFace utils
sys.path.append('../../HuggingFace') # Include HuggingFace
from NNDF.logger import G_LOGGER

# ONNX conversion script
try:
    from GPT3.convert_te_onnx_to_trt_onnx import replace_customop_qdq_with_onnx_qdq
except:
    G_LOGGER.error("Opset19 conversion script needs to be copied to GPT3 folder before running this script.")

# Set polygraphy logging level here.
PG_LOGGER.module_severity = PG_LOGGER.INFO

class MegatronGPTSingleInputExportableModel(MegatronGPTExportableModel):
    """
    Wrapper for MegatronGPTExportableModel to export ONNX with a single input
    """

    def __init__(self, model, max_seq_len):
        super().__init__(model)
        self.cfg = model.cfg
        self.max_seq_len = max_seq_len

    def forward(self, tokens):
        def model_forward(tokens):
            position_ids, attention_mask = self.get_position_ids_and_mask(tokens, self.max_seq_len)
            assert tokens.shape == position_ids.shape
            assert attention_mask.shape[2] == attention_mask.shape[3] == tokens.shape[1] == position_ids.shape[1]
            return self.model.forward(
                tokens=tokens.cuda(),
                text_position_ids=position_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                labels=None,
            )

        with torch.no_grad(), torch.inference_mode(), torch.autocast(
            'cuda', dtype=self.dtype
        ), warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning, module=r'.*')
            if self.fp8_enabled:
                with transformer_engine.pytorch.onnx_export(self.fp8_enabled), transformer_engine.pytorch.fp8_autocast(
                    enabled=self.fp8_enabled, fp8_recipe=self.fp8_recipe
                ):
                    output_tensor = model_forward(tokens)
            else:
                output_tensor = model_forward(tokens)
        return output_tensor

    def get_position_ids_and_mask(self, data, max_seq_len):
        seq_len = data.size()[1]
        # Attention mask (lower triangular).
        attention_mask = torch.tril(torch.ones(
            (1, max_seq_len, max_seq_len), device=data.device)).view(
                1, 1, max_seq_len, max_seq_len)

        # Position ids.
        position_ids = torch.arange(max_seq_len, dtype=torch.long,
                                    device=data.device)
        position_ids = position_ids[:seq_len].unsqueeze(0).expand_as(data)

        # Convert attention mask to binary:
        attention_mask = (attention_mask < 0.5)

        return position_ids, attention_mask[:1, :1, :seq_len, :seq_len]

    def input_example(self):
        ids = self.model.tokenizer.text_to_ids("how is the weather on Sunday morning?")
        id_tensors = torch.unsqueeze(torch.LongTensor(ids), dim=0)
        G_LOGGER.debug(f"Calling input_example shape {id_tensors.shape}")
        return id_tensors, # return a tuple

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    def input_names(self) -> List[str]:
        return ['input_ids']

def get_trtexec_cmd(onnx_fpath, cfg, bs):
    max_seq_len = cfg.model.max_seq_len
    trtexec_cmd = f"trtexec --onnx={onnx_fpath}"
    min_shapes = f"--minShapes=input_ids:{bs}x1"
    opt_shapes = f"--optShapes=input_ids:{bs}x{max_seq_len // 2}"
    max_shapes = f"--maxShapes=input_ids:{bs}x{max_seq_len}"
    if not cfg.use_one_input:
        min_shapes += f",position_ids:{bs}x1"
        opt_shapes += f",position_ids:{bs}x{max_seq_len // 2}"
        max_shapes += f",position_ids:{bs}x{max_seq_len}"
    if not cfg.trt_engine_options.use_fp8:
        min_shapes += ",attention_mask:1x1x1x1"
        opt_shapes += f",attention_mask:1x1x{max_seq_len // 2}x{max_seq_len // 2}"
        max_shapes += f",attention_mask:1x1x{max_seq_len}x{max_seq_len}"

    if cfg.use_cache:
        trtexec_cmd += " --profile=0"
        nbheads, headsize = cfg.model.nb_heads, cfg.model.head_size
        input_k = get_past_key_name('*')
        input_v = get_past_value_name('*')
        # ("sequence", "batch", nbheads, headsize)
        min_shapes += f",{input_k}:0x{bs}x{nbheads}x{headsize},{input_v}:0x{bs}x{nbheads}x{headsize}"
        opt_shapes += f",{input_k}:0x{bs}x{nbheads}x{headsize},{input_v}:0x{bs}x{nbheads}x{headsize}"
        max_shapes += f",{input_k}:0x{bs}x{nbheads}x{headsize},{input_v}:0x{bs}x{nbheads}x{headsize}"
    trtexec_cmd += f" {min_shapes} {opt_shapes} {max_shapes}"

    if cfg.use_cache:
        trtexec_cmd += " --profile=1"

        min_shapes = f"--minShapes=input_ids:{bs}x1"
        opt_shapes = f"--optShapes=input_ids:{bs}x1"
        max_shapes = f"--maxShapes=input_ids:{bs}x1"
        if not cfg.use_one_input:
            min_shapes += f",position_ids:{bs}x1"
            opt_shapes += f",position_ids:{bs}x1"
            max_shapes += f",position_ids:{bs}x1"
        if not cfg.trt_engine_options.use_fp8:
            min_shapes += ",attention_mask:1x1x1x1"
            opt_shapes += f",attention_mask:1x1x{max_seq_len // 2}x{max_seq_len // 2}"
            max_shapes += f",attention_mask:1x1x{max_seq_len}x{max_seq_len}"

        nbheads, headsize = cfg.model.nb_heads, cfg.model.head_size
        input_k = get_past_key_name('*')
        input_v = get_past_value_name('*')
        # ("sequence", "batch", nbheads, headsize)
        min_shapes += f",{input_k}:1x{bs}x{nbheads}x{headsize},{input_v}:1x{bs}x{nbheads}x{headsize}"
        opt_shapes += f",{input_k}:{max_seq_len // 2}x{bs}x{nbheads}x{headsize},{input_v}:{max_seq_len // 2}x{bs}x{nbheads}x{headsize}"
        max_shapes += f",{input_k}:{max_seq_len - 1}x{bs}x{nbheads}x{headsize},{input_v}:{max_seq_len - 1}x{bs}x{nbheads}x{headsize}"
        trtexec_cmd += f" {min_shapes} {opt_shapes} {max_shapes}"

    use_tf32 = cfg.trt_engine_options.use_tf32
    use_bf16 = cfg.trt_engine_options.use_bf16
    use_fp16 = cfg.trt_engine_options.use_fp16
    use_fp8 = cfg.trt_engine_options.use_fp8
    use_bf16 = cfg.trt_engine_options.use_bf16
    trtexec_cmd += " --noTF32" if not use_tf32 else ""
    trtexec_cmd += " --fp16" if use_fp16 else ""
    trtexec_cmd += " --bf16" if use_bf16 else ""
    trtexec_cmd += " --fp8" if use_fp8 else ""
    trtexec_cmd += " --bf16" if use_bf16 else ""
    trtexec_cmd += " --timingCacheFile=functional.cache --preview=+fasterDynamicShapes0805,+disableExternalTacticSourcesForCore0805"
    return trtexec_cmd


# Use ONNX graphsurgeon to add KV-cache to ONNX file
# Reusing the HF demo names.
def get_past_key_name(layer_id):
    past_key_name = f"past_key_values.{layer_id}.decoder.key"
    return past_key_name

def get_past_value_name(layer_id):
    past_value_name = f"past_key_values.{layer_id}.decoder.value"
    return past_value_name

def get_past_shape(nbheads, headsize):
    return ("sequence_past_decoder_length", "batch", nbheads, headsize)

def get_present_key_name(layer_id: int):
    present_key_name = f"present_key_values.{layer_id}.decoder.key"
    return present_key_name

def get_present_value_name(layer_id: int):
    present_value_name = f"present_key_values.{layer_id}.decoder.value"
    return present_value_name

def get_present_shape(nbheads, headsize):
    return ("sequence_present_decoder_length", "batch", nbheads, headsize)

def get_new_key_name(layer_id: int):
    new_key_name = f"new_key_values.{layer_id}.decoder.key"
    return new_key_name

def get_new_value_name(layer_id: int):
    new_value_name = f"new_key_values.{layer_id}.decoder.value"
    return new_value_name

def get_new_shape(nbheads, headsize):
    return ("sequence", "batch", nbheads, headsize)

def add_kvcache_for(g, layer_id, qkv_split, nbheads, headsize, dtype, kv_output_policy):
    query_new, key_new, value_new = qkv_split.outputs
    key_consumers = [c for c in key_new.outputs]
    value_consumers = [c for c in value_new.outputs]

    def add_graph_past_inputs():
        past_key = gs.Variable(
            name=get_past_key_name(layer_id),
            dtype=dtype,
            shape=get_past_shape(nbheads, headsize))
        past_value = gs.Variable(
            name=get_past_value_name(layer_id),
            dtype=dtype,
            shape=get_past_shape(nbheads, headsize))
        g.inputs.append(past_key)
        g.inputs.append(past_value)
        return past_key, past_value

    def add_concat(concat_name, input0, input1, output_name):
        concat_out = gs.Variable(
            output_name,
            dtype=dtype,
            shape=get_present_shape(nbheads, headsize))

        concat = gs.Node(op="Concat", name=concat_name,
            inputs=[input0, input1], outputs=[concat_out],
            attrs={"axis": 0})
        g.nodes.append(concat)
        return concat_out

    def add_cache_outputs(kv_output_policy):
        if kv_output_policy == "kv_cache_concat":
            g.outputs.append(key_concat_out)
            g.outputs.append(value_concat_out)
        elif kv_output_policy == "kv_new":
            key_new.dtype = dtype
            key_new.shape = get_new_shape(nbheads, headsize)
            key_new.name = get_new_key_name(layer_id)
            value_new.dtype = dtype
            value_new.shape = get_new_shape(nbheads, headsize)
            value_new.name = get_new_value_name(layer_id)
            g.outputs.append(key_new)
            g.outputs.append(value_new)
        else:
            raise ValueError(f"Unsupported kv_output_policy: {kv_output_policy}")

    past_key, past_value = add_graph_past_inputs()
    key_concat_out = add_concat(f"key.{layer_id}.concat",
        past_key, key_new, get_present_key_name(layer_id))
    value_concat_out = add_concat(f"value.{layer_id}.concat",
        past_value, value_new, get_present_value_name(layer_id))

    for c in key_consumers:
        c.inputs[0] = key_concat_out
    for c in value_consumers:
        c.inputs[0] = value_concat_out

    add_cache_outputs(kv_output_policy)


def add_kvcache(g, nbheads, headsize, dtype, kv_output_policy):
    """Add KV-cache to each Transformer layer's QKV split """
    qkv_split_nodes = [node for node in g.nodes if node.op == "Split"]
    G_LOGGER.debug(f"Found {len(qkv_split_nodes)} QKV-split nodes")

    for layer_id, qkv_split in enumerate(qkv_split_nodes):
        add_kvcache_for(g, layer_id, qkv_split, nbheads, headsize, dtype, kv_output_policy)
    G_LOGGER.debug("Done adding cache operations")
    return len(qkv_split_nodes)


def normalize_dyn_axes_to_hf_names(g, vocab_size):
    g.inputs[0].name = "input_ids"
    g.inputs[0].shape = ("batch", "sequence")
    if len(g.inputs) > 1:
        g.inputs[1].name = "position_ids"
        g.inputs[1].shape = ("batch", "sequence")
    g.outputs[0].name = "logits"
    g.outputs[0].shape = ("batch", "sequence", vocab_size)
    G_LOGGER.debug("Done normalizing dynamic axes names to HuggingFace demo names")


def process_onnx(
    kv_output_policy,
    onnx_input_fpath,
    onnx_output_fpath,
    separate_param_files,
    nbheads, headsize, vocab_size, dtype):
    """
    Process an ONNX model, add KV cache inputs and output, save result model to a specified path.
    """
    G_LOGGER.info(f"Importing {onnx_input_fpath}... this will take some time")
    g = gs.import_onnx(onnx.load(onnx_input_fpath))
    normalize_dyn_axes_to_hf_names(g, vocab_size)
    num_layers = add_kvcache(g, nbheads, headsize, dtype, kv_output_policy)

    g.cleanup().toposort()
    G_LOGGER.info(f"Exporting {onnx_output_fpath}")
    model = gs.export_onnx(g)
    G_LOGGER.info(f"Saving {onnx_output_fpath}")
    if separate_param_files:
        onnx.save_model(model, onnx_output_fpath, save_as_external_data=True,
             all_tensors_to_one_file = False, convert_attribute=False)
    else:
        onnx.save_model(model, onnx_output_fpath, save_as_external_data=False)
    G_LOGGER.info(f"Done: {onnx_output_fpath}")
    return num_layers


def create_dir_if_not_exist(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir) and dir != "":
        G_LOGGER.info(f"Making directory {dir}")
        os.makedirs(dir)


class NeMoConverter():
    """
    A class to convert a NeMo model to an ONNX file, and convert an ONNX file to a TensorRT engine.
    """
    def __init__(self, cfg, model_type=ModelPT):
        self.model_type = model_type
        self.cfg = cfg
        self.model = None

    def nemo_to_onnx(self) -> str:
        """
        Convert a NeMo model to an ONNX model, return the file path to the ONNX model.
        """
        if self.model == None:
            self.model = load_nemo_model(self.cfg, self.model_type)

        if not isinstance(self.model, Exportable):
            G_LOGGER.error("Your NeMo model class ({}) is not Exportable.".format(model.__class__.__name__))
            sys.exit(1)

        if hasattr(self.model.cfg, "fp8") and self.model.cfg.fp8 == True:
            if self.cfg.trt_engine_options.use_fp8 == False:
                G_LOGGER.info("Turning on trt_engine_options.use_fp8 because NeMo model is in FP8 precision.")
                self.cfg.trt_engine_options.use_fp8 = True
        else:
            if self.cfg.trt_engine_options.use_fp8 == True:
                G_LOGGER.info("Turning off trt_engine_options.use_fp8 because NeMo model is not in FP8 precision.")
                self.cfg.trt_engine_options.use_fp8 = False

        onnx_out = self.cfg.onnx_model_file
        create_dir_if_not_exist(onnx_out)
        check_trace = self.cfg.export_options.runtime_check
        onnx_names = []

        dynamic_axes={
            'input_ids': {0: "batch", 1: "sequence"},
            'position_ids': {0: "batch", 1: "sequence"},
            'logits': {0: "batch", 1: "sequence"},
        }

        if self.cfg.use_one_input:
            # Use a wrapper class to get rid of inputs other than input_ids.
            self.model = MegatronGPTSingleInputExportableModel(self.model, self.cfg.model.max_seq_len)
            del dynamic_axes['position_ids']

        try:
            self.model.to(device=self.cfg.export_options.device).freeze()
            self.model.eval()
            if not self.cfg.trt_engine_options.use_fp8:
                G_LOGGER.info("Exporting ONNX with attention_mask")
                dynamic_axes['attention_mask'] = {2: "sequence", 3: "sequence"}

            if self.cfg.trt_engine_options.use_fp8:
                G_LOGGER.info(f"Setting max sequence length to {self.cfg.model.max_seq_len}")
                os.environ['NVTE_ONNX_KVCACHE_max_seq_len'] = str(self.cfg.model.max_seq_len)

            self.model.export(
                onnx_out,
                onnx_opset_version=self.cfg.export_options.onnx_opset,
                do_constant_folding=self.cfg.export_options.do_constant_folding,
                dynamic_axes=dynamic_axes,
                check_trace=check_trace,
                check_tolerance=self.cfg.export_options.check_tolerance,
                verbose=self.cfg.export_options.verbose,
            )
            onnx_names = [augment_filename(onnx_out, subnet_name) for subnet_name in self.model.list_export_subnets()]

        except Exception as e:
            G_LOGGER.error(
                "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                    self.model.__class__
                )
            )
            raise e

        release_nemo_model(self.model)
        assert len(onnx_names) == 1
        os.rename(onnx_names[0], onnx_out)
        return onnx_out


    def create_kv_onnx(self, onnx_input_fpath, onnx_output_fpath, kv_output_policy="kv_new"):
        """
        Create an ONNX model with KV cache from `onnx_input_fpath`, save the ONNX model to `onnx_output_fpath`.
        No return value for this function.
        """
        assert os.path.splitext(onnx_input_fpath)[1] == ".onnx", "Input ONNX file must end with '.onnx'."
        assert os.path.splitext(onnx_output_fpath)[1] == ".onnx", "Output ONNX file must end with '.onnx'."

        nbheads, headsize = self.cfg.model.nb_heads, self.cfg.model.head_size
        dtype = onnx.TensorProto.BFLOAT16 if self.cfg.trt_engine_options.use_bf16 else onnx.TensorProto.FLOAT16
        assert nbheads * headsize == self.cfg.model.hidden_size, "Model hidden size does not match."
        num_qkvs = process_onnx(kv_output_policy,
            onnx_input_fpath, onnx_output_fpath, separate_param_files=True,
            nbheads=nbheads, headsize=headsize, vocab_size=self.cfg.model.vocab_size, dtype=dtype)

        G_LOGGER.info(f"Number of QKV subgraphs = {num_qkvs}, number of layers = {self.cfg.model.num_layers}")
        if num_qkvs != self.cfg.model.num_layers:
            raise ValueError("Number of QKV subgraphs must be the same as number of layers in the model.")
        G_LOGGER.info(f"Saved KV-cache onnx to {onnx_output_fpath}.")


    # Reads an onnx file and creates a trt engine file
    def onnx_to_trt(self, onnx_fpath, trt_fpath):
        """
        Convert an ONNX model from `onnx_fpath` to a TensorRT engine, and save the result to `trt_fpath`.
        """
        # Set up polygraphy config
        use_tf32 = self.cfg.trt_engine_options.use_tf32
        use_fp16 = self.cfg.trt_engine_options.use_fp16
        use_fp8 = self.cfg.trt_engine_options.use_fp8
        use_bf16 = self.cfg.trt_engine_options.use_bf16

        # Create optimization profiles
        bs = self.cfg.batch_size
        max_seq_len = self.cfg.model.max_seq_len
        profile_non_kv = Profile()
        profile_non_kv.add(name="input_ids", min=(bs, 1), opt=(bs, max_seq_len // 2), max=(bs, max_seq_len)) # (batch, sequence)
        if not self.cfg.use_one_input:
            profile_non_kv.add(name="position_ids", min=(bs, 1), opt=(bs, max_seq_len // 2), max=(bs, max_seq_len)) # (batch, sequence)
            # For FP8 precision, attention mask is created inside transformer_engine.
            if not self.cfg.trt_engine_options.use_fp8:
                profile_non_kv.add(name="attention_mask", min=(1, 1, 1, 1), opt=(1, 1, max_seq_len // 2, max_seq_len // 2), max=(1, 1, max_seq_len, max_seq_len)) # (1, 1, sequence, sequence)

        num_layers, nbheads, headsize = self.cfg.model.num_layers, self.cfg.model.nb_heads, self.cfg.model.head_size
        if self.cfg.use_cache:
            for i in range(num_layers):
                input_k = get_past_key_name(i)
                input_v = get_past_value_name(i)
                # (sequence, batch, nbheads, headsize)
                profile_non_kv.add(name=input_k, min=(0, bs, nbheads, headsize), opt=(0, bs, nbheads, headsize), max=(0, bs, nbheads, headsize))
                profile_non_kv.add(name=input_v, min=(0, bs, nbheads, headsize), opt=(0, bs, nbheads, headsize), max=(0, bs, nbheads, headsize))

        profiles = [profile_non_kv]

        # When enabling KV-cache, use first profile for context phase and second profile for generation phase
        if self.cfg.use_cache:
            profile_kv = Profile()
            profile_kv.add(name="input_ids", min=(bs, 1), opt=(bs, 1), max=(bs, 1)) # (batch, sequence)
            if not self.cfg.use_one_input:
                profile_kv.add(name="position_ids", min=(bs, 1), opt=(bs, 1), max=(bs, 1)) # (batch, sequence)
                # For FP8 precision, attention mask is created inside transformer_engine.
                if not self.cfg.trt_engine_options.use_fp8:
                    profile_kv.add(name="attention_mask", min=(1, 1, 1, 1), opt=(1, 1, max_seq_len // 2, max_seq_len // 2), max=(1, 1, max_seq_len, max_seq_len)) # (1, 1, sequence, sequence)

            assert num_layers > 0
            nbheads, headsize = self.cfg.model.nb_heads, self.cfg.model.head_size
            for i in range(num_layers):
                input_k = get_past_key_name(i)
                input_v = get_past_value_name(i)
                # (sequence, batch, nbheads, headsize)
                profile_kv.add(name=input_k, min=(1, bs, nbheads, headsize), opt=(max_seq_len // 2, bs, nbheads, headsize), max=(max_seq_len-1, bs, nbheads, headsize))
                profile_kv.add(name=input_v, min=(1, bs, nbheads, headsize), opt=(max_seq_len // 2, bs, nbheads, headsize), max=(max_seq_len-1, bs, nbheads, headsize))
            profiles = [profile_kv, profile_non_kv]


        # Read about these arguments here:
        # https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/backend/trt/config.py
        # Note that the precision args below *enable*, not *require*, the specified precision
        preview_features = [PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805,
                            PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]
        trt_config = CreateConfig(
            tf32=use_tf32,
            fp16=use_fp16,
            bf16=use_bf16,
            profiles=profiles,
            precision_constraints="obey",
            preview_features=preview_features,
            fp8=use_fp8,
            load_timing_cache=self.cfg.trt_engine_options.timing_cache,
        )

        # Print out trtexec command for debugging
        G_LOGGER.debug(" >>> trtexec command for debugging:")
        G_LOGGER.debug(get_trtexec_cmd(onnx_fpath, self.cfg, bs))

        G_LOGGER.info(f"Reading ONNX file at {onnx_fpath}")
        network = network_from_onnx_path(onnx_fpath)
        G_LOGGER.info("Building TRT engine")
        engine = engine_from_network(network, config=trt_config)
        G_LOGGER.info(f"Saving TRT engine to {trt_fpath}")
        save_engine(engine, trt_fpath)


    @staticmethod
    def get_opset19_onnx_fpath(onnx_fpath) -> str:
        suffix = '.opset19.onnx'
        return os.path.join(os.path.split(onnx_fpath)[0], os.path.splitext(os.path.split(onnx_fpath)[1])[0] + suffix)


    @staticmethod
    def onnx_to_opset19(onnx_fpath) -> str:
        """
        Convert a ONNX model `onnx_fpath` to be with standard opset19 Q/DQ nodes, return a string
        contains a file path to the result ONNX if any conversion is performed, otherwise return `None`.
        """
        mappings = replace_customop_qdq_with_onnx_qdq([onnx_fpath], os.path.split(onnx_fpath)[0],
                                                   create_netron_compatible_model=False,
                                                   remove_cast_before_q=False,
                                                   remove_cast_after_dq=False,
                                                   change_qdq_scale_precision="")
        if (not mappings) or (onnx_fpath not in mappings) or (mappings[onnx_fpath] == None):
            G_LOGGER.error(f"Opset19 onnx file conversion failed for {onnx_fpath}.")
            assert False

        G_LOGGER.info(f"Converted {onnx_fpath} to {mappings[onnx_fpath]} for opset19.")
        return mappings[onnx_fpath]

@hydra_runner(config_path="./", config_name="megatron_gpt_demo")
def main(cfg):
    assert cfg.onnx_model_file != None and cfg.trt_engine_file != None
    create_dir_if_not_exist(cfg.trt_engine_file)

    # Convert NeMo model to ONNX model
    converter = NeMoConverter(cfg, MegatronGPTModel)
    onnx_name = converter.nemo_to_onnx()
    G_LOGGER.info(f"Using intermediate onnx file path {onnx_name}")

    # Convert Q/DQ nodes to use standard opset19 operators
    op19_onnx = NeMoConverter.onnx_to_opset19(onnx_name)
    if op19_onnx != None:
        G_LOGGER.info(f"Get opset19 onnx file {op19_onnx}")
        onnx_name = op19_onnx

    # Add KV cache to ONNX model
    if cfg.use_cache:
        G_LOGGER.info(f"Converting {onnx_name} with KV-cache support")
        kv_output_policy = "kv_new"
        new_dir = str(os.path.dirname(onnx_name)) + f"_{kv_output_policy}"
        onnx_output_fpath = os.path.join(new_dir, onnx_name.split("/")[-1])
        create_dir_if_not_exist(onnx_output_fpath)
        converter.create_kv_onnx(onnx_name, onnx_output_fpath, kv_output_policy)
        onnx_name = onnx_output_fpath

    # Convert ONNX model to TRT engine
    trt_fpath = cfg.trt_engine_file
    converter.onnx_to_trt(onnx_name, trt_fpath)

if __name__ == '__main__':
    main()
