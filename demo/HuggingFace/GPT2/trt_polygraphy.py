# std
import os
from re import S
import sys
import logging
from typing import Dict, List, Tuple

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)


# numpy
import numpy as np

# torch
import torch

# huggingface
from transformers import GPT2Tokenizer, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin

# TRT-HuggingFace
from interface import PolygraphyCommand
from networks import (
    NNFolderWorkspace,
    NetworkMetadata,
    NetworkModels,
    NetworkModel,
    NetworkResult,
    NetworkRuntime,
    Precision,
    TimingProfile,
)

from polygraphy_utils import TRTRunner
from GPT2.frameworks import GPT2HuggingFace
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from GPT2.measurements import gpt2_inference, full_inference_greedy
from GPT2.export import GPT2ONNXFile, GPT2TRTEngine

class TRTHFRunner(TRTRunner, GenerationMixin):
    """Runner that adds interop support for HF and HF provided greedy_search functions."""
    def __init__(self, engine_fpath: str, network_metadata: NetworkMetadata, hf_config: PretrainedConfig):
        super().__init__(engine_fpath, network_metadata)
        self.config = hf_config

class GPT2TRTDecoder(TRTHFRunner):

    def prepare_inputs_for_generation(self, input_ids, **kwargs):  
        # Todo (@pchadha): add position_ids, token_type_ids support
        return {
            "input_ids": input_ids,
        }

    def forward(self, input_ids, **kwargs):
        if not isinstance(input_ids, np.ndarray):
            assert isinstance(input_ids, torch.Tensor)
            input_ids = input_ids.cpu().numpy().astype(np.int32)
 
        logits = self.trt_context.infer(
            {"input_ids": input_ids}
        )["logits"]

        return CausalLMOutputWithCrossAttentions(logits=torch.from_numpy(logits).to("cuda"))
        
class GPT2Polygraphy(PolygraphyCommand):
    def __init__(self):
        super().__init__(
            GPT2ModelTRTConfig, "Runs polygraphy results for GPT2 model.", GPT2HuggingFace
        )
        self.gpt2_trt = None

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_trt_engine: bool = False,
        keep_onnx_model: bool = False,
        keep_torch_model: bool = False,
    ) -> None:
        # Deactivates context
        if self.gpt2_trt is not None:
            self.gpt2_trt.release()

        if not keep_trt_engine:
            self.gpt2_engine.cleanup()

        self.frameworks_cmd.cleanup(workspace, keep_onnx_model, keep_torch_model)

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Dict[str, NetworkModel],
        inference_input: str,
        timing_profile: TimingProfile,
    ) -> NetworkResult:

        tokenizer = GPT2Tokenizer.from_pretrained(metadata.variant)
        input_ids = tokenizer(inference_input, return_tensors="pt").input_ids

        # get single decoder iteration inference timing profile 
        _, decoder_e2e_median_time = gpt2_inference(
            self.gpt2_trt, input_ids, timing_profile
        )

        # get complete decoder inference result and its timing profile 
        sample_output, full_e2e_median_runtime = full_inference_greedy(
            self.gpt2_trt, input_ids, timing_profile,
            max_length=GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
        )

        semantic_outputs = []
        for i, sample_output in enumerate(sample_output):
            semantic_outputs.append(tokenizer.decode(sample_output, skip_special_tokens=True))

        return NetworkResult(
            output_tensor=sample_output,
            semantic_output=semantic_outputs,
            median_runtime=[
                NetworkRuntime(
                    name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    runtime=decoder_e2e_median_time,
                ),
                NetworkRuntime(
                    name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    runtime=full_e2e_median_runtime,
                ),
            ],
            models=NetworkModels(
                torch=None,
                onnx=list(onnx_fpaths.values()),
                trt=[
                    NetworkModel(
                        name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                        fpath=self.gpt2_engine.fpath,
                    ),
                ],
            ),
        )

    def run_polygraphy(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Tuple[NetworkModel],
        network_input: List[str],
        working_directory: str,
        keep_trt_engine: bool,
        keep_onnx_model: bool,
        keep_torch_model: bool,
        timing_profile: TimingProfile,
    ) -> List[NetworkResult]:
        workspace = NNFolderWorkspace(
            self.frameworks_cmd.config.network_name, metadata, working_directory
        )

        results = []
        try:
            # no fpath provided for onnx files, download them
            if len(onnx_fpaths) == 0:
                onnx_fpaths = self.frameworks_cmd.generate_and_download_framework(
                    metadata, workspace
                ).onnx
            else:
                keep_onnx_model = True
                keep_torch_model = True

            # Output networks shall not exceed number of network segments explicitly defined by configuraiton file.
            assert len(onnx_fpaths) == len(
                GPT2ModelTRTConfig.NETWORK_SEGMENTS
            ), "There should only be {} exported ONNX segments in GPT2 model."

            hash_onnx_fpath = {v.name: v for v in onnx_fpaths}

            gpt2_onnx_fpath = hash_onnx_fpath[
                GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
            ].fpath

            self.gpt2_engine = GPT2ONNXFile(gpt2_onnx_fpath, metadata).as_trt_engine(gpt2_onnx_fpath + ".engine")
            tfm_config = GPT2Config(
                use_cache=metadata.other.kv_cache,
            )
            self.gpt2_trt = GPT2TRTDecoder(self.gpt2_engine.fpath, metadata, tfm_config)

            for ninput in network_input:
                results.append(
                    self.execute_inference(
                        metadata, hash_onnx_fpath, ninput, timing_profile
                    )
                )

        finally:
            self.cleanup(workspace, keep_trt_engine, keep_onnx_model, keep_torch_model)

        return results

    def add_args(self, parser) -> None:
        # use the same args as frameworks.py
        self.frameworks_cmd.add_args(parser)
        polygraphy_group = parser.add_argument_group("polygraphy")
        polygraphy_group.add_argument(
            "--onnx-fpath",
            default=None,
            help="Path to GPT2 ONNX model. If None is supplied, scripts will generate them from HuggingFace.",
        )
        polygraphy_group.add_argument(
            "--fp16", action="store_true", help="Enables fp16 TensorRT tactics."
        )
        polygraphy_group.add_argument(
            "--save-trt-engine",
            action="store_true",
            help="Saves TensorRT runtime engine in working directory.",
        )

    def args_to_network_models(self, args) -> List[NetworkModel]:
        gpt2_fpath_check = args.onnx_fpath is None

        network_models = None
        if gpt2_fpath_check:
            network_models = tuple()
        else:
            onnx_decoder = NetworkModel(
                name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=args.onnx_fpath,
            )
            network_models = (onnx_decoder)

        return network_models

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        frameworks_parsed_metadata = self.frameworks_cmd.args_to_network_metadata(args)

        return NetworkMetadata(
            variant=frameworks_parsed_metadata.variant,
            precision=Precision(fp16=args.fp16),
            other=frameworks_parsed_metadata.other,
        )


RUN_CMD = GPT2Polygraphy()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
