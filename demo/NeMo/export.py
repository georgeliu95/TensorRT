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

import logging
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
from nemo_utils import load_nemo_model, release_nemo_model
from nemo.utils.export_utils import augment_filename
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTExportableModel

# onnx
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnx_graphsurgeon as gs
    
# polygraphy
from polygraphy.backend.trt import Profile, CreateConfig, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.logger import G_LOGGER

# tensorrt
from tensorrt import PreviewFeature
import torch
import transformer_engine

# Set logging level here.
G_LOGGER.module_severity = G_LOGGER.INFO

class MegatronGPTSingleInputExportableModel(MegatronGPTExportableModel):
    """
    Wrapper for MegatronGPTExportableModel to export ONNX with a single input
    """

    def __init__(self, model):
        super().__init__(model, max_seq_len)
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
        print(f"Calling input_example shape {id_tensors.shape}")
        return id_tensors, # return a tuple

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    def input_names(self) -> List[str]:
        return ['input_ids']


def nemo_to_onnx(cfg):
    model = load_nemo_model(cfg, ModelPT)
    if not isinstance(model, Exportable):
        logging.error("Your NeMo model class ({}) is not Exportable.".format(model.__class__.__name__))
        sys.exit(1)

    if hasattr(model.cfg, "fp8") and model.cfg.fp8 == True:
        if cfg.trt_engine_options.use_fp8 == False:
            logging.info("Turning on trt_engine_options.use_fp8 because NeMo model is in FP8 precision.")
            cfg.trt_engine_options.use_fp8 = True
    else:
        if cfg.trt_engine_options.use_fp8 == True:
            logging.info("Turning off trt_engine_options.use_fp8 because NeMo model is not in FP8 precision.")
            cfg.trt_engine_options.use_fp8 = False

    onnx_out = cfg.onnx_model_file
    check_trace = cfg.export_options.runtime_check
    onnx_names = []

    dynamic_axes={
        'input_ids': {0: "batch", 1: "sequence"},
        'position_ids': {0: "batch", 1: "sequence"},
        'logits': {0: "batch", 1: "sequence"},
    }

    if cfg.use_one_input:
        # Use a wrapper class to get rid of inputs other than input_ids.
        model = MegatronGPTSingleInputExportableModel(model, cfg.model.max_seq_len)
        del dynamic_axes['position_ids']

    try:
        model.to(device=cfg.export_options.device).freeze()
        model.eval()
        if not cfg.trt_engine_options.use_fp8:
            logging.info("Exporting ONNX with attention_mask")
            dynamic_axes['attention_mask'] = {2: "sequence", 3: "sequence"}

        if cfg.trt_engine_options.use_fp8:
            logging.info(f"Setting max sequence length to {cfg.model.max_seq_len}")
            os.environ['NVTE_ONNX_KVCACHE_max_seq_len'] = str(cfg.model.max_seq_len)

        model.export(
            onnx_out,
            onnx_opset_version=cfg.export_options.onnx_opset,
            do_constant_folding=cfg.export_options.do_constant_folding,
            dynamic_axes=dynamic_axes,
            check_trace=check_trace,
            check_tolerance=cfg.export_options.check_tolerance,
            verbose=cfg.export_options.verbose,
        )
        onnx_names = [augment_filename(onnx_out, subnet_name) for subnet_name in model.list_export_subnets()]

    except Exception as e:
        logging.error(
            "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                model.__class__
            )
        )
        raise e

    release_nemo_model(model)
    return onnx_names

def get_trtexec_cmd(onnx_fpath, cfg, bs, num_layers):
    # Print out trtexec command for debugging
    trtexec_cmd = f"trtexec --onnx={onnx_fpath}"
    min_shapes = "--minShapes=input_ids:1x1,position_ids:1x1"
    opt_shapes = "--optShapes=input_ids:1x1,position_ids:1x1"
    max_shapes = f"--maxShapes=input_ids:{bs}x256,position_ids:{bs}x256"
    if not cfg.trt_engine_options.use_fp8:
        min_shapes += ",attention_mask:1x1x1x1"
        opt_shapes += ",attention_mask:1x1x1x1"
        max_shapes += ",attention_mask:1x1x256x256"

    if cfg.enable_kv_cache:
        nbheads, headsize = cfg.model.nb_heads, cfg.model.head_size
        for i in range(num_layers):
            input_k = get_past_key_name(i)
            input_v = get_past_value_name(i)
            # ("sequence", "batch", nbheads, headsize)
            min_shapes += f",{input_k}:0x1x{nbheads}x{headsize},{input_v}:0x1x{nbheads}x{headsize}"
            opt_shapes += f",{input_k}:1x1x{nbheads}x{headsize},{input_v}:1x1x{nbheads}x{headsize}"
            max_shapes += f",{input_k}:256x{bs}x{nbheads}x{headsize},{input_v}:256x{bs}x{nbheads}x{headsize}"

    use_tf32 = cfg.trt_engine_options.use_tf32
    use_fp16 = cfg.trt_engine_options.use_fp16
    use_fp8 = cfg.trt_engine_options.use_fp8
    trtexec_cmd += f" {min_shapes} {opt_shapes} {max_shapes}"
    trtexec_cmd += " --noTF32" if not use_tf32 else ""
    trtexec_cmd += " --fp16" if use_fp16 else ""
    trtexec_cmd += " --fp8" if use_fp8 else ""
    trtexec_cmd += " --timingCacheFile=functional.cache --preview=+fasterDynamicShapes0805,+disableExternalTacticSourcesForCore0805"
    return trtexec_cmd

# Reads an onnx file and creates a trt engine file
def onnx_to_trt(cfg, onnx_fpath, num_layers=0):
    trt_fpath = cfg.trt_engine_file

    # Set up polygraphy config
    use_tf32 = cfg.trt_engine_options.use_tf32
    use_fp16 = cfg.trt_engine_options.use_fp16
    use_fp8 = cfg.trt_engine_options.use_fp8
    if cfg.trt_engine_options.use_bf16:
        raise ValueError("bf16 isn't supported by polygraphy yet")

    # Create optimization profiles
    bs = cfg.batch_size
    max_seq_len = cfg.model.max_seq_len
    profile_non_kv = Profile()
    profile_non_kv.add(name="input_ids", min=(bs, 1), opt=(bs, max_seq_len // 2), max=(bs, max_seq_len)) # (batch, sequence)
    if not cfg.use_one_input:
        profile_non_kv.add(name="position_ids", min=(bs, 1), opt=(bs, max_seq_len // 2), max=(bs, max_seq_len)) # (batch, sequence)
        # For FP8 precision, attention mask is created inside transformer_engine.
        if not cfg.trt_engine_options.use_fp8:
            profile_non_kv.add(name="attention_mask", min=(1, 1, 1, 1), opt=(1, 1, max_seq_len // 2, max_seq_len // 2), max=(1, 1, max_seq_len, max_seq_len)) # (1, 1, sequence, sequence)

    if cfg.enable_kv_cache:
        assert num_layers > 0
        nbheads, headsize = cfg.model.nb_heads, cfg.model.head_size
        for i in range(num_layers):
            input_k = get_past_key_name(i)
            input_v = get_past_value_name(i)
            # (sequence, batch, nbheads, headsize)
            profile_non_kv.add(name=input_k, min=(0, bs, nbheads, headsize), opt=(0, bs, nbheads, headsize), max=(0, bs, nbheads, headsize))
            profile_non_kv.add(name=input_v, min=(0, bs, nbheads, headsize), opt=(0, bs, nbheads, headsize), max=(0, bs, nbheads, headsize))

    profiles = [profile_non_kv]

    # When enabling KV-cache, use first profile for context phase and second profile for generation phase
    if cfg.enable_kv_cache:
        profile_kv = Profile()
        profile_kv.add(name="input_ids", min=(bs, 1), opt=(bs, 1), max=(bs, 1)) # (batch, sequence)
        if not cfg.use_one_input:
            profile_kv.add(name="position_ids", min=(bs, 1), opt=(bs, 1), max=(bs, 1)) # (batch, sequence)
            # For FP8 precision, attention mask is created inside transformer_engine.
            if not cfg.trt_engine_options.use_fp8:
                profile_kv.add(name="attention_mask", min=(1, 1, 1, 1), opt=(1, 1, max_seq_len // 2, max_seq_len // 2), max=(1, 1, max_seq_len, max_seq_len)) # (1, 1, sequence, sequence)

        assert num_layers > 0
        nbheads, headsize = cfg.model.nb_heads, cfg.model.head_size
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
        profiles=profiles,
        precision_constraints="obey",
        preview_features=preview_features,
        fp8=use_fp8,
        load_timing_cache=cfg.trt_engine_options.timing_cache,
    )

    # Print out trtexec command for debugging
    logging.debug(" >>> trtexec command for debugging:")
    logging.debug(get_trtexec_cmd(onnx_fpath, cfg, bs, num_layers))

    logging.info(f"Reading ONNX file at {onnx_fpath}")
    network = network_from_onnx_path(onnx_fpath)
    logging.info("Building TRT engine")
    engine = engine_from_network(network, config=trt_config)
    logging.info(f"Saving TRT engine to {trt_fpath}")
    save_engine(engine, trt_fpath)

# Use ONNX graphsurgeon to add KV-cache to ONNX file
# Reusing the HF demo names.
def get_past_key_name(layer_id: int):
    past_key_name = f"past_key_values.{layer_id}.decoder.key"
    return past_key_name

def get_past_value_name(layer_id: int):
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
    print(f"Found {len(qkv_split_nodes)} QKV-split nodes")

    for layer_id, qkv_split in enumerate(qkv_split_nodes):
        add_kvcache_for(g, layer_id, qkv_split, nbheads, headsize, dtype, kv_output_policy)
    print("Done adding cache operations")
    return len(qkv_split_nodes)


def normalize_dyn_axes_to_hf_names(g):
    g.inputs[0].name = "input_ids"
    g.inputs[0].shape = ("batch", "sequence")
    if len(g.inputs) > 1:
        g.inputs[1].name = "position_ids"
        g.inputs[1].shape = ("batch", "sequence")
    g.outputs[0].name = "logits"
    g.outputs[0].shape = ("batch", "sequence", 50304)
    print("Done normalizing dynamic axes names to HuggingFace demo names")


def process_onnx(
    kv_output_policy,
    onnx_input_fpath,
    onnx_output_fpath,
    separate_param_files,
    nbheads, headsize, dtype):
    print(f"Importing {onnx_input_fpath}... this will take some time")
    g = gs.import_onnx(onnx.load(onnx_input_fpath))
    normalize_dyn_axes_to_hf_names(g)
    num_layers = add_kvcache(g, nbheads, headsize, dtype, kv_output_policy)

    g.cleanup().toposort()
    print(f"Exporting {onnx_output_fpath}")
    model = gs.export_onnx(g)
    print(f"Saving {onnx_output_fpath}")
    if separate_param_files:
        onnx.save_model(model, onnx_output_fpath, save_as_external_data=True,
             all_tensors_to_one_file = False, convert_attribute=False)
    else:
        onnx.save_model(model, onnx_output_fpath, save_as_external_data=False)
    print(f"Done: {onnx_output_fpath}")
    return num_layers


def create_kv_onnx(cfg, onnx_fpath):
    assert os.path.splitext(onnx_fpath)[1] == ".onnx", "ONNX file must end with '.onnx'."
    kv_output_policy = "kv_new"
    onnx_inp_fpath = onnx_fpath

    dir = str(os.path.dirname(onnx_fpath)) + f"_{kv_output_policy}"
    onnx_outp_fpath = os.path.join(dir, onnx_fpath.split("/")[-1])
    create_dir_if_not_exist(onnx_outp_fpath)

    nbheads, headsize = cfg.model.nb_heads, cfg.model.head_size
    dtype = np.float16
    assert nbheads * headsize == cfg.model.hidden_size, "Model hidden size does not match."
    logging.info(f"Converting onnx from {onnx_inp_fpath} to {onnx_outp_fpath}.")
    num_layers = process_onnx(kv_output_policy,
        onnx_inp_fpath, onnx_outp_fpath, separate_param_files=True,
        nbheads=nbheads, headsize=headsize, dtype=dtype)
    return onnx_outp_fpath, num_layers

def create_dir_if_not_exist(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir) and dir != "":
        logging.info(f"Making directory {dir}")
        os.makedirs(dir)

def find_node_by_type(graph, search_tensor, is_node_input, search_node_type=None):
    for idx, node in enumerate(graph.node):
        search_container = node.output
        if is_node_input:
            search_container = node.input
        for node_tensor in search_container:
            if search_node_type and node.op_type != search_node_type:
                continue
            if node_tensor == search_tensor:
                return node, idx
    return None, None

def redirect_quantize_input(graph, q_node):
    assert(q_node.op_type == 'QuantizeLinear')
    q_input = q_node.input[0]
    cast_node, cast_node_idx = find_node_by_type(graph, q_input, False, 'Cast')
    if cast_node:
        q_node.input[0] = cast_node.input[0]
        return [cast_node_idx]
    return []

def redirect_dequantize_output(graph, dq_node):
    assert(dq_node.op_type == 'DequantizeLinear')
    dq_output = dq_node.output[0]
    cast_node, cast_node_idx = find_node_by_type(graph, dq_output, True, 'Cast')
    if cast_node:
        dq_node.output[0] = cast_node.output[0]
        return [cast_node_idx]
    return []

def get_attr_numpy_tensor(attr):
    assert(attr.type == onnx.AttributeProto.TENSOR)
    return numpy_helper.to_array(attr.t)

def get_attr(node, search_attr_name):
    for idx, attr in enumerate(node.attribute):
        if attr.name == search_attr_name:
            return attr, idx
    return None, None

def cast_scale(graph, qdq_node, cast_to):
    assert(cast_to in ['fp32', 'fp16'])
    assert(qdq_node.op_type in ['QuantizeLinear', 'DequantizeLinear'])
    constant_node_idx = None
    scale_tensor = qdq_node.input[1]
    constant_node, constant_node_idx = find_node_by_type(graph, scale_tensor, False, 'Constant')
    scale_cast_to_dtype = None
    onnx_cast_to_dtype = None
    if cast_to == 'fp16':
        scale_cast_to_dtype = np.dtype(np.float32)
        onnx_cast_to_dtype = onnx.TensorProto.FLOAT16
    elif cast_to == 'fp32':
        scale_cast_to_dtype = np.dtype(np.float32)
        onnx_cast_to_dtype = onnx.TensorProto.FLOAT

    if constant_node:
        scale_attr, _ = get_attr(constant_node, 'value')
        assert(scale_attr)
        numpy_scale = get_attr_numpy_tensor(scale_attr)
        if numpy_scale.dtype != scale_cast_to_dtype:
            logging.debug(f'Change {qdq_node.name} scale from {numpy_scale.dtype} to {scale_cast_to_dtype}')
            numpy_scale = numpy_scale.astype(scale_cast_to_dtype)
            tensor_name = constant_node.name + '_casted'
            create_constant_tensor(graph, tensor_name, onnx_cast_to_dtype, numpy_scale)
            qdq_node.input[1] = tensor_name
    else:
        logging.warning(f'No constant node connected to {qdq_node} as scale')

    if constant_node_idx:
        return [constant_node_idx]
    return []

def create_constant_tensor(graph, name, dtype, np_tensor):
    tensor_value_info = helper.make_tensor_value_info(name, dtype, np_tensor.shape)
    graph.input.append(tensor_value_info)
    helper.make_tensor(name, data_type=dtype, dims=(), vals=[0])

    tensor_initializer = helper.make_tensor(name, dtype, np_tensor.shape, np_tensor.flatten().tolist())
    graph.initializer.append(tensor_initializer)


def custom_op_to_opset19(graph, node, use_int32_quantization, remove_cast_before_q, remove_cast_after_dq, change_qdq_scale_precision):
    """
    Convert custom operators to opset19
    """
    assert(node.op_type in ['TRT_FP8QuantizeLinear', 'TRT_FP8DequantizeLinear'])
    is_dq = node.op_type == 'TRT_FP8DequantizeLinear'
    logging.debug(f'Convert {node.name} to Opset19')
    orig_node_name = node.name
    new_node_name = orig_node_name + '_converted'

    quant_to = TensorProto.FLOAT8E4M3FN
    if use_int32_quantization:
        quant_to = TensorProto.INT32

    # Add zero point to the node
    tensor_name = new_node_name + '_zero_point'
    create_constant_tensor(graph, tensor_name, quant_to, np.array([0]))
    node.input.append(tensor_name)

    node.op_type = "QuantizeLinear"
    node_idxs_to_delete = []
    if is_dq:
        node.op_type = "DequantizeLinear"
        if remove_cast_after_dq:
            node_idxs_to_delete += redirect_dequantize_output(graph, node)
            if change_qdq_scale_precision:
                node_idxs_to_delete += cast_scale(graph, node, change_qdq_scale_precision)
    else:
        if remove_cast_before_q:
            node_idxs_to_delete += redirect_quantize_input(graph, node)
            if change_qdq_scale_precision:
                node_idxs_to_delete += cast_scale(graph, node, change_qdq_scale_precision)

    node.name = new_node_name
    logging.debug(f'Convert Done\n')
    return node_idxs_to_delete

def check_model(graph):
    """
    Check if a model needs to be converted or not
    """
    converted_qdq_ops = ['TRT_FP8QuantizeLinear', 'TRT_FP8DequantizeLinear']
    passed_check = True
    for node in graph.node:
        if node.op_type in converted_qdq_ops:
            logging.error(f'Node \"{node.name}\" of type {node.op_type} should have been removed')
            passed_check = False
    return passed_check

def replace_customop_qdq_with_onnx_qdq(te_onnx_file, results_path, create_netron_compatible_model, remove_cast_before_q, remove_cast_after_dq, change_qdq_scale_precision):
    """
    Convert custom TRT nodes to standard ONNX Q/DQ nodes
    """
    model = onnx.load(te_onnx_file, load_external_data=False)
    model.opset_import[0].version = 19
    graph = model.graph
    converted_qdq_ops = ['TRT_FP8QuantizeLinear', 'TRT_FP8DequantizeLinear']

    try:
        node_idxs_to_delete = []
        converted = False
        for node in graph.node:
            if node.op_type in converted_qdq_ops:
                converted = True
                node_idxs_to_delete += custom_op_to_opset19(graph, node, create_netron_compatible_model, remove_cast_before_q, remove_cast_after_dq, change_qdq_scale_precision)

        if converted:
            assert(check_model(graph))
            node_idxs_to_delete = reversed(sorted(node_idxs_to_delete))
            for node_idx in node_idxs_to_delete:
                del(graph.node[node_idx])
            suffix = '.opset19'
            if create_netron_compatible_model:
                suffix += '.netron'
            suffix += '.onnx'
            new_model_filename = os.path.join(results_path, os.path.splitext(os.path.split(te_onnx_file)[1])[0] + suffix)
            onnx.save_model(model, new_model_filename)
            logging.info(f'The converted model is saved at {new_model_filename}!')
            return new_model_filename
        else:
            logging.info(f'No conversion was done with {te_onnx_file}!')
    except Exception as ex:
        logging.error(f'Failed: {ex}')
    return None

def onnx_to_opset19(onnx_fpath):
    return replace_customop_qdq_with_onnx_qdq(onnx_fpath, os.path.split(onnx_fpath)[0],
                                              create_netron_compatible_model=False,
                                              remove_cast_before_q=False,
                                              remove_cast_after_dq=False,
                                              change_qdq_scale_precision="")

@hydra_runner(config_path="./", config_name="megatron_gpt_demo")
def main(cfg):
    assert cfg.onnx_model_file != None and cfg.trt_engine_file != None
    create_dir_if_not_exist(cfg.onnx_model_file)
    create_dir_if_not_exist(cfg.trt_engine_file)

    # Convert NeMo model to ONNX model
    onnx_names = nemo_to_onnx(cfg)
    assert len(onnx_names) == 1
    logging.info(f"Using intermediate onnx file path {onnx_names[0]}")

    onnx_name = onnx_names[0]

    # Convert Q/DQ nodes to use standard opset19 operators
    op19_onnx = onnx_to_opset19(onnx_name)
    if op19_onnx != None:
        logging.info(f"Get opset19 onnx file {op19_onnx}")
        onnx_name = op19_onnx

    num_layers = 0
    if cfg.enable_kv_cache:
        logging.info(f"Converting {onnx_name} with KV-cache support")
        onnx_name, num_layers = create_kv_onnx(cfg, onnx_name)
    onnx_to_trt(cfg, onnx_name, num_layers)

if __name__ == '__main__':
    main()
