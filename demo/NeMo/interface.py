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
import random
import sys
import time
from typing import List, Union

from cuda import cuda
from tqdm import tqdm
import numpy as np
import torch

from transformers import PretrainedConfig
from omegaconf import OmegaConf, listconfig

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from GPT3.decoding import full_inference
from GPT3.GPT3ModelConfig import GPT3ModelTRTConfig
from GPT3.lambada_dataset import Lambada
from GPT3.nemo_utils import get_computeprob_response
from GPT3.sequence_perplexity import SequencePerplexity

sys.path.append('../HuggingFace') # Include HuggingFace
from NNDF.general_utils import NNFolderWorkspace
from NNDF.logger import G_LOGGER
from NNDF.networks import (
    DeprecatedCache,
    Precision,
    NetworkMetadata,
    TimingProfile,
    BenchmarkingResult,
    NetworkResult,
    NetworkCheckpointResult,
)
from NNDF.interface import NetworkCommand

# Manually set by referring to examples/nlp/language_modeling/conf/megatron_gpt_config.yaml
# If a field cannot be found, set to None.
DEFAULT_CONFIG = {
    "is_encoder_decoder": False,
    "is_decoder": True,
    "architectures": [ "GPT3NeMoModel" ],
}

GPT3CONFIG_MAPPINGS = {
    "gpt-126m": PretrainedConfig.from_dict(dict({"_name_or_path": "gpt-126m",
        "fp8_path": None,
        "fp16_path": None,
        "bf16_path": None,
        "num_heads": 12,
        "num_layers": 12,
        "hidden_size": 768,
        "max_position_embeddings": 2048,
    }, **DEFAULT_CONFIG)),
    "gpt-1.3b": PretrainedConfig.from_dict(dict({"_name_or_path": "gpt-1.3b",
        "fp8_path": None,
        "fp16_path": "/home/fp8gpt3/nemo_models/nemo_gpt1.3B_fp16.nemo",
        "bf16_path": None,
        "num_heads": 16,
        "num_layers": 24,
        "hidden_size": 2048,
        "max_position_embeddings": 2048,
    }, **DEFAULT_CONFIG)),
    "gpt-5b": PretrainedConfig.from_dict(dict({"_name_or_path": "gpt-5b",
        "fp8_path": "/home/fp8gpt3/nemo_models/nemo_gpt5B_fp8_tp1.nemo",
        "fp16_path": "/home/fp8gpt3/nemo_models/nemo_gpt5B_fp16_tp1.nemo",
        "bf16_path": None,
        "num_heads": 32,
        "num_layers": 24,
        "hidden_size": 4096,
        "max_position_embeddings": 2048,
    }, **DEFAULT_CONFIG)),
}

def load_dataset(dataset_name, base_dir):
    ds_map = {"Lambada": Lambada(base_dir)}
    return ds_map[dataset_name]

def get_accuracy_metric(cfg):
    topN = [int(i.strip()) for i in cfg.top_n.split(",")]
    m_map = {"Perplexity": SequencePerplexity(topN)}
    return m_map[cfg.metric]

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

class NeMoCommand(NetworkCommand):
    def __init__(
        self,
        nemo_cfg,
        config_class,
        description,
        **kwargs
    ):
        self.nemo_cfg = nemo_cfg
        super().__init__(config_class, description, **kwargs)

    def validate_and_set_precision(self, fp8, fp16, bf16):
        if fp8:
            if fp16:
                G_LOGGER.info("Use FP8-FP16 precision.")
            if bf16:
                G_LOGGER.info("Use FP8-BF16 precision.")
        elif fp16:
            G_LOGGER.info("Use pure FP16 precision.")
        elif bf16:
            G_LOGGER.info("Use pure BF16 precision.")
        else:
            fp16 = True
            G_LOGGER.warn("Precision is not specified. Use pure FP16 precision by default.")

        self.fp8, self.fp16, self.bf16 = fp8, fp16, bf16
        self.nemo_cfg.trt_engine_options.use_fp8 = fp8
        self.nemo_cfg.trt_engine_options.use_fp16 = fp16
        self.nemo_cfg.trt_engine_options.use_bf16 = bf16
        if fp16:
            self.nemo_cfg.trainer.precision = "16"
        elif bf16:
            self.nemo_cfg.trainer.precision = "bf16"
        else:
            self.nemo_cfg.trainer.precision = "32"

    def update_hyperparams(self, model_config):
        self.nemo_cfg.model.num_layers = model_config.num_layers
        self.nemo_cfg.model.nb_heads = model_config.num_heads
        self.nemo_cfg.model.head_size = model_config.hidden_size // model_config.num_heads
        self.nemo_cfg.model.hidden_size = model_config.hidden_size
        self.nemo_cfg.model.encoder_seq_length = model_config.max_position_embeddings
        self.nemo_cfg.model.max_position_embeddings = model_config.max_position_embeddings

    def setup_environment(
        self,
        variant: str,
        working_dir: str = "temp",
        batch_size: int = 1,
        num_beams: int = 1,
        use_cache: bool = True,
        verbose: bool = False,
        info: bool = False,
        iterations: int = None,
        warmup: int = None,
        number: int = None,
        duration: int = None,
        percentile: int = None,
        benchmarking_mode: bool = False,
        cleanup: bool = False,
        action: str = None,
        max_seq_len: int = None,
        fp8: bool = True,
        fp16: bool = False,
        bf16: bool = False,
        input_seq_len: int = None,
        output_seq_len: int = None,
        nemo_model: str = None,
        **kwargs,
    ) -> None:
        """
        Use Arguments from command line or user specified to setup config for the model.
        """
        self.validate_and_set_precision(fp8, fp16, bf16)

        if not torch.cuda.is_available():
            raise EnvironmentError("GPU is required for NeMo demo.")

        # Initialize CUDA Driver API
        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda initialization failed with error: {}".format(err))

        # See https://pytorch.org/docs/stable/_modules/torch.html#set_float32_matmul_precision
        torch.set_float32_matmul_precision('medium')

        assert action != None, "Action must be specified"
        if action == "accuracy":
            self.nemo_cfg.mode = "accuracy"
            self.nemo_cfg.inference.compute_logprob = True
            self.nemo_cfg.inference.all_probs = True
            self.nemo_cfg.inference.greedy = True
            self.nemo_cfg.inference.add_BOS = False
            self.nemo_cfg.inference.tokens_to_generate = 1
            self.nemo_cfg.inference.min_tokens_to_generate = 0
            self.nemo_cfg.inference.temperature = 1.0
            self.nemo_cfg.inference.top_k = 0
            self.nemo_cfg.inference.top_p = 0.9
            self.nemo_cfg.inference.repetition_penalty = 1.0
        elif action == "benchmark":
            self.nemo_cfg.mode = "benchmark"
            if input_seq_len != None:
                self.nemo_cfg.benchmark.input_seq_len = input_seq_len
            if output_seq_len != None:
                self.nemo_cfg.benchmark.output_seq_len = output_seq_len
            self.nemo_cfg.inference.tokens_to_generate = self.nemo_cfg.benchmark.output_seq_len
            self.nemo_cfg.inference.min_tokens_to_generate = self.nemo_cfg.benchmark.output_seq_len

        if max_seq_len != None:
            self.nemo_cfg.model.max_seq_len = max_seq_len

        # Get NeMo model path
        assert variant in GPT3CONFIG_MAPPINGS
        model_config = GPT3CONFIG_MAPPINGS[variant]
        if self.fp8 == True:
            self.nemo_cfg.gpt_model_file = model_config.fp8_path
            self.nemo_cfg.trt_engine_options.use_fp8 = True
        elif self.fp16 == True:
            self.nemo_cfg.gpt_model_file = model_config.fp16_path
            self.nemo_cfg.trt_engine_options.use_fp16 = True
        elif self.bf16 == True:
            self.nemo_cfg.gpt_model_file = model_config.bf16_path
            self.nemo_cfg.trt_engine_options.use_bf16 = True
        else:
            raise ValueError("A precision needs to be set to retrieve a model.")

        if nemo_model != None:
            self.nemo_cfg.gpt_model_file = nemo_model

        self.nemo_cfg.batch_size = batch_size
        self.nemo_cfg.use_cache = use_cache

        if self.nemo_cfg.gpt_model_file == None:
            G_LOGGER.error("No model exists based on specified precisions.")
            raise ValueError("Model not found.")

        self.update_hyperparams(model_config)

        # HuggingFace code
        if verbose:
            G_LOGGER.setLevel(level=G_LOGGER.DEBUG)
        elif info:
            G_LOGGER.setLevel(level=G_LOGGER.INFO)

        if variant is None:
            G_LOGGER.error("You should specify --variant to run NeMo demo")
            return

        if self._args is not None:
            G_LOGGER.info("Setting up environment with arguments: {}".format(self._args))
        else:
            G_LOGGER.info("User-customized API is called")

        self.metadata = NetworkMetadata(
            variant=variant,
            precision=Precision(fp16=self.fp16),
            use_cache=use_cache,
            num_beams=num_beams,
            batch_size=batch_size,
            other=DeprecatedCache(kv_cache=use_cache)
        )

        self.config = self.config_class(
            metadata = self.metadata
        )

        self.config.from_nemo_config(self.nemo_cfg)
        self.benchmarking_mode = benchmarking_mode

        self.workspace = NNFolderWorkspace(
            self.config, working_dir
        )

        self.timing_profile = TimingProfile(
            iterations=iterations,
            number=number,
            warmup=warmup,
            duration=duration,
            percentile=percentile,
        )

        self.keep_torch_model = not cleanup
        self.keep_onnx_model = not cleanup
        self.keep_trt_engine = not cleanup

        self.process_framework_specific_arguments(**kwargs)

    def process_framework_specific_arguments(self, **kwargs):
        pass

    def run(self) -> Union[List[NetworkResult], BenchmarkingResult]:
        """
        Main entry point of our function which compiles and generates our model data for command-line mode.
        The general process for the commands are all the same:
        (1) Download the model
        (2) Run either checkpoint or benchmark
        (3) Returns the result
        """
        t0 = time.time()
        self.models = self.setup_tokenizer_and_model()
        t1 = time.time()
        G_LOGGER.info("setup_tokenizer_and_model() takes {:.4f}s in total.".format(t1 - t0))

        results = []
        ppl = None
        random.seed(self.nemo_cfg.inference.seed)
        np.random.seed(self.nemo_cfg.inference.seed)
        torch.manual_seed(self.nemo_cfg.inference.seed)
        if self.nemo_cfg.mode == "accuracy":
            G_LOGGER.debug("Run in accuracy mode.")
            eval_ppl = get_accuracy_metric(self.nemo_cfg.accuracy)
            dataset = load_dataset(self.nemo_cfg.accuracy.dataset, self.workspace.rootdir)
            tokenizer = self.tokenizer

            def eval_ppl_with_batch_input(eval_ppl, batch_input):
                ds_input = dataset.preprocess_input(tokenizer, batch_input)

                response = full_inference(
                    model=self.model,
                    inputs=ds_input.inputs,
                    cfg=self.nemo_cfg,
                )
                response = get_computeprob_response(tokenizer, response, ds_input.inputs)
                eval_ppl.update(ds_input=ds_input, response=response, tokenizer=tokenizer)

            batch_input = []
            for doc in tqdm(dataset.load()):
                batch_input.append(doc)

                if len(batch_input) == self.nemo_cfg.batch_size:
                    eval_ppl_with_batch_input(eval_ppl, batch_input)
                    batch_input.clear()
            if len(batch_input):
                eval_ppl_with_batch_input(eval_ppl, batch_input)

            ppl, _, acc_text = eval_ppl.compute()
            print("***************************")
            print("{} ppl: {}, {}".format(self.nemo_cfg.accuracy.dataset, ppl, acc_text))
            print("***************************")
        elif self.nemo_cfg.mode == "benchmark":
            G_LOGGER.debug("Run in benchmark mode.")
            rand_input = get_random_input(self.model.tokenizer, self.nemo_cfg.batch_size, self.nemo_cfg.benchmark.input_seq_len, self.nemo_cfg.benchmark.output_seq_len)

            for _ in range(self.timing_profile.warmup):
                output = full_inference(self.model, rand_input, self.nemo_cfg)

            times = []
            for _ in range(self.timing_profile.iterations):
                start_time = time.perf_counter()
                output = full_inference(self.model, rand_input, self.nemo_cfg)
                times.append(time.perf_counter() - start_time)

            print("***************************")
            total_time = sum(times)
            avg_time = total_time / float(self.timing_profile.iterations)
            times.sort()
            percentile95 = times[int(self.timing_profile.iterations * 0.95)]
            percentile99 = times[int(self.timing_profile.iterations * 0.99)]
            throughput = float(self.nemo_cfg.batch_size * self.nemo_cfg.benchmark.output_seq_len) / avg_time
            print("Running {:} iterations with batch size: {:}, input sequence length: {:} and output sequence length: {:}".format(self.timing_profile.iterations, self.nemo_cfg.batch_size, self.nemo_cfg.benchmark.input_seq_len, self.nemo_cfg.benchmark.output_seq_len))
            print("  Total Time: {:0.5f} s, Average Time: {:0.5f} s, 95th Percentile Time: {:0.5f} s, 99th Percentile Time: {:0.5f} s, Throughput: {:0.2f} tokens/s".format(total_time, avg_time, percentile95, percentile99, throughput))
            print("***************************")
        else:
            G_LOGGER.debug("Run in inference mode.")
            assert self.nemo_cfg.mode == "inference"
            if self.nemo_cfg.runtime == 'nemo' and hasattr(self.model.cfg, "fp8") and self.model.cfg.fp8 == True and self.nemo_cfg.batch_size % 8 != 0:
                new_batch_size = ((self.nemo_cfg.batch_size + 7) // 8) * 8
                print("Update batch size from {} to {} for NeMo FP8 inference.".format(self.nemo_cfg.batch_size, new_batch_size))
                self.nemo_cfg.batch_size = new_batch_size

            nb_paddings = 0
            while (len(self.nemo_cfg.prompts) % self.nemo_cfg.batch_size) != 0:
                self.nemo_cfg.prompts.append(self.nemo_cfg.prompts[-1])
                nb_paddings += 1

            batch_idx = 0
            start = 0
            while True:
                inputs = OmegaConf.to_container(listconfig.ListConfig(self.nemo_cfg.prompts[start:start+self.nemo_cfg.batch_size]))
                output = full_inference(self.model, inputs, self.nemo_cfg)
                output = remove_padded_prompts(output, nb_paddings)
                print("***************************")
                print("Batch {}: {}".format(batch_idx, output))
                print("***************************")
                batch_idx += 1
                start += self.nemo_cfg.batch_size
                if start >= len(self.nemo_cfg.prompts):
                    break

        t2 = time.time()
        G_LOGGER.info("Inference session is {:.4f}s in total.".format(t2 - t1))

        # Release runtime objects
        if self.nemo_cfg.runtime == 'onnx':
            del self.model.onnx
        elif self.nemo_cfg.runtime == 'trt':
            del self.model.trt

        return results, ppl

    def add_args(self) -> None:
        general_group = self._parser.add_argument_group("general")
        general_group.add_argument(
            "--help",
            "-h",
            help="Shows help message for NeMo commands.",
            action="store_true",
        )
        general_group.add_argument(
            "--verbose", "-v",
            help="Display verbose logs.",
            action="store_true"
        )
        general_group.add_argument(
            "--info", help="Display info logs.", action="store_true"
        )
        general_group.add_argument(
            "--working-dir", "-wd",
            help="Location of where to save the model and other downloaded files.",
            required=True,
        )

        timing_group = self._parser.add_argument_group("inference measurement")
        timing_group.add_argument(
            "--iterations",
            type=int,
            help="Number of iterations to measure.",
            default=NetworkCommand.DEFAULT_ITERATIONS,
        )
        timing_group.add_argument(
            "--warmup",
            type=int,
            help="Number of warmup iterations before actual measurement occurs.",
            default=NetworkCommand.DEFAULT_WARMUP,
        )

        model_config_group = self._parser.add_argument_group("model")
        model_config_group.add_argument(
            "--nemo-model",
            help="Set a NeMo model to be used.",
            type=str,
            default=None
        )
        model_config_group.add_argument(
            "--max-seq-len",
            help="Set maximum sequence lengths used for a GPT model.",
            type=int,
            default=None,
        )
        model_config_group.add_argument(
            "--batch-size", "-b",
            help="Chosen batch size for given network",
            required=False,
            type=int,
            default=1
        )
        model_config_group.add_argument(
            "--variant", "-m",
            help="model to generate",
            required=True,
            choices=GPT3ModelTRTConfig.TARGET_MODELS,
        )
        model_config_group.add_argument(
            "--use-cache",
            "-kv",
            help="Enable KV cache",
            action="store_true",
            default=False,
        )
        model_config_group.add_argument(
            "--fp8",
            action="store_true",
            help="Use FP8 precision.",
            default=False
        )
        model_config_group.add_argument(
            "--fp16",
            action="store_true",
            help="Uses fp16 tactics.",
            default=False
        )
        model_config_group.add_argument(
            "--bf16",
            action="store_true",
            help="Use BF16 precision.",
            default=False
        )

    def __call__(self):
        t0 = time.time()
        self.add_args()
        self._args = self._parser.parse_args()
        if "help" in self._args and self._args.help == True:
            self._parser.print_help()
            exit(0)

        self.setup_environment(
            **vars(self._args),
            benchmarking_mode=(self._args.action == "benchmark"),
        )
        t1 = time.time()
        G_LOGGER.info("Set up environment takes {:.4f}s.".format(t1 - t0))

        network_results, ppl_results = self.run()
        return NetworkCheckpointResult(
            network_results=network_results,
            accuracy=0,
            perplexity=0,
            models=self.models
        )
