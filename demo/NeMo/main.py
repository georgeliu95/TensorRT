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
This is the script to run GPT text generation.

Usage:
    a. run greedy inference from a nemo file:
        python main.py gpt_model_file=<PATH_TO_MODEL> prompts=[prompt1,prompt2]

    b. run accuracy check using sequence perplexity:
        python main.py gpt_model_file=<PATH_TO_MODEL> mode=accuracy

    c. run with TensorRT:
        python main.py runtime=trt trt_engine_file=<PATH_TO_TRT_ENGINE>

    d. run with ONNX Runtime:
        python main.py runtime=onnx onnx_model_file=<PATH_TO_ONNX>
"""

from cuda import cuda
import numpy as np
import random
import time
import torch
from megatron.core import parallel_state
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.modules.common.text_generation_utils import get_computeprob_response
from omegaconf import OmegaConf, listconfig
from sequence_perplexity import SequencePerplexity
from nemo.core.config import hydra_runner
from lambada_dataset import Lambada
from tqdm import tqdm

from decoding import full_inference
from nemo_utils import load_nemo_model


def load_dataset(dataset_name):
    ds_map = {"Lambada": Lambada()}
    return ds_map[dataset_name]


def get_accuracy_metric(metric_name):
    m_map = {"Perplexity": SequencePerplexity()}
    return m_map[metric_name]

def remove_padded_prompts(output, nb_paddings):
    if nb_paddings == 0:
        return output
    result = {}
    for k, v in output.items():
        if v != None and (type(v) is list or type(v) is torch.Tensor):
            v = v[:-nb_paddings]
        result[k] = v
    return result

def get_random_input(tokenizer, batch_size, in_seq_len, out_seq_len):
    vocab_size = tokenizer.tokenizer.vocab_size
    return (torch.randint(0, vocab_size, (batch_size, in_seq_len + out_seq_len), dtype=torch.int64).cuda(),
            (torch.ones(batch_size, dtype=torch.int64) * in_seq_len).cuda())

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None
    def forward(self, x):
        raise Exception("BaseModel forward method is not intended to be called.")

@hydra_runner(config_path="./", config_name="megatron_gpt_demo")
def main(cfg) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for the inference")

    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Cuda initialization failed with error: {}".format(err))

    # See https://pytorch.org/docs/stable/_modules/torch.html#set_float32_matmul_precision
    torch.set_float32_matmul_precision('medium')

    if cfg.mode == "accuracy":
        cfg.inference.compute_logprob = True
        cfg.inference.all_probs = True
        cfg.inference.greedy = True
        cfg.inference.add_BOS = False
        cfg.inference.tokens_to_generate = 1
        cfg.inference.min_tokens_to_generate = 0
        cfg.inference.temperature = 1.0
        cfg.inference.top_k = 0
        cfg.inference.top_p = 0.9
        cfg.inference.repetition_penalty = 1.0
    elif cfg.mode == "benchmark":
        cfg.inference.tokens_to_generate = cfg.benchmark.output_seq_len
        cfg.inference.min_tokens_to_generate = cfg.benchmark.output_seq_len

    if cfg.runtime == 'nemo':
        model = load_nemo_model(cfg)
    else:
        model = BaseModel()
        model.cfg = cfg.model
        model.tokenizer = get_tokenizer(tokenizer_name='megatron-gpt-345m', vocab_file=None, merges_file=None)
        if cfg.runtime == 'onnx':
            from onnx_utils import load_onnx_model
            model.onnx = load_onnx_model(cfg)
        elif cfg.runtime == 'trt':
            from trt_utils import load_trt_model
            model.trt = load_trt_model(cfg)
        else:
            raise Exception(f"Runtime {cfg.runtime} is not supported")

    random.seed(cfg.inference.seed)
    np.random.seed(cfg.inference.seed)
    if cfg.mode == "accuracy":
        eval_ppl = get_accuracy_metric(cfg.accuracy.metric)
        dataset = load_dataset(cfg.accuracy.dataset)
        tokenizer = model.tokenizer

        def eval_ppl_with_batch_input(eval_ppl, batch_input):
            ds_input = dataset.preprocess_input(tokenizer, batch_input)

            response = full_inference(
                model=model,
                inputs=ds_input.inputs,
                cfg=cfg,
            )
            response = get_computeprob_response(tokenizer, response, ds_input.inputs)
            eval_ppl.update(ds_input=ds_input, response=response, tokenizer=tokenizer)

        batch_input = []
        for doc in tqdm(dataset.load()):
            batch_input.append(doc)

            if len(batch_input) == cfg.batch_size:
                eval_ppl_with_batch_input(eval_ppl, batch_input)
                batch_input.clear()
        if len(batch_input):
            eval_ppl_with_batch_input(eval_ppl, batch_input)

        print("***************************")
        print("{} ppl: {}".format(cfg.accuracy.dataset, eval_ppl.compute()))
        print("***************************")
    elif cfg.mode == "benchmark":
        rand_input = get_random_input(model.tokenizer, cfg.batch_size, cfg.benchmark.input_seq_len, cfg.benchmark.output_seq_len)

        for _ in range(cfg.benchmark.warm_up):
            output = full_inference(model, rand_input, cfg)

        times = []
        for _ in range(cfg.benchmark.loop):
            start_time = time.perf_counter()
            output = full_inference(model, rand_input, cfg)
            times.append(time.perf_counter() - start_time)

        print("***************************")
        total_time = sum(times)
        avg_time = total_time / float(cfg.benchmark.loop)
        times.sort()
        percentile95 = times[int(cfg.benchmark.loop * 0.95)]
        percentile99 = times[int(cfg.benchmark.loop * 0.99)]
        throughput = float(cfg.batch_size * cfg.benchmark.output_seq_len) / avg_time
        print("Running {:} iterations with batch size: {:}, input sequence length: {:} and output sequence length: {:}".format(cfg.benchmark.loop, cfg.batch_size, cfg.benchmark.input_seq_len, cfg.benchmark.output_seq_len))
        print("  Total Time: {:0.5f} s, Average Time: {:0.5f} s, 95th Percentile Time: {:0.5f} s, 99th Percentile Time: {:0.5f} s, Throughput: {:0.2f} tokens/s".format(total_time, avg_time, percentile95, percentile99, throughput))
        print("***************************")
    else:
        assert cfg.mode == "inference"
        if cfg.runtime == 'nemo' and hasattr(model.cfg, "fp8") and model.cfg.fp8 == True and cfg.batch_size % 8 != 0:
            new_batch_size = ((cfg.batch_size + 7) // 8) * 8
            print("Update batch size from {} to {} for NeMo FP8 inference.".format(cfg.batch_size, new_batch_size))
            cfg.batch_size = new_batch_size

        nb_paddings = 0
        while (len(cfg.prompts) % cfg.batch_size) != 0:
            cfg.prompts.append("")
            nb_paddings += 1

        batch_idx = 0
        start = 0
        while True:
            inputs = OmegaConf.to_container(listconfig.ListConfig(cfg.prompts[start:start+cfg.batch_size]))
            output = full_inference(model, inputs, cfg)
            output = remove_padded_prompts(output, nb_paddings)
            print("***************************")
            print("Batch {}: {}".format(batch_idx, output))
            print("***************************")
            batch_idx += 1
            start += cfg.batch_size
            if start >= len(cfg.prompts):
                break

    # Release runtime objects
    if cfg.runtime == 'onnx':
        del model.onnx
    elif cfg.runtime == 'trt':
        del model.trt

if __name__ == '__main__':
    main()
