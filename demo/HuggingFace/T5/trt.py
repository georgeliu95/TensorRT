# std
import argparse
import os
from re import S
import sys
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
from transformers import T5Tokenizer, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin

# TRT-HuggingFace
from NNDF.interface import TRTInferenceCommand
from NNDF.networks import (
    NetworkMetadata,
    NetworkModels,
    NetworkModel,
    NetworkResult,
    NetworkRuntime,
    Precision,
    TimingProfile,
)

from NNDF.polygraphy_utils import TRTRunner
from NNDF.general_utils import NNFolderWorkspace
from T5.frameworks import T5FHuggingFace
from T5.T5ModelConfig import T5ModelTRTConfig
from T5.measurements import decoder_inference, encoder_inference, full_inference_greedy
from T5.export import T5DecoderONNXFile, T5EncoderONNXFile


class TRTHFRunner(TRTRunner, GenerationMixin):
    """Runner that adds interop support for HF and HF provided greedy_search functions."""

    def __init__(
        self,
        engine_fpath: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
    ):
        super().__init__(engine_fpath, network_metadata)
        self.config = hf_config


class T5TRTEncoder(TRTHFRunner):
    """TRT implemented network interface that can be used to measure inference time."""

    def forward(self, input_ids, *args, **kwargs):
        if not isinstance(input_ids, np.ndarray):
            input_ids = input_ids.numpy().astype(np.int32)

        return self.trt_context.infer({"input_ids": input_ids})["hidden_states"]


class T5TRTDecoder(TRTHFRunner):
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "encoder_hidden_states": kwargs["encoder_hidden_states"],
        }

    def forward(self, input_ids, encoder_hidden_states, *args, **kwargs):
        if not isinstance(input_ids, np.ndarray):
            input_ids = input_ids.numpy().astype(np.int32)
        if not isinstance(encoder_hidden_states, np.ndarray):
            encoder_hidden_states = encoder_hidden_states.numpy().astype(np.float32)

        input_sequence_shape = input_ids.shape[1]
        encoder_sequence_shape = encoder_hidden_states.shape[1]
        if encoder_sequence_shape != input_sequence_shape:
            new_encoder_hidden_states = np.zeros(
                (
                    encoder_hidden_states.shape[0],
                    input_sequence_shape,
                    encoder_hidden_states.shape[2],
                ),
                dtype=np.float32,
            )
            encoder_broadcast = min(input_sequence_shape, encoder_sequence_shape)
            new_encoder_hidden_states[:, :encoder_broadcast, :] = encoder_hidden_states[
                :, :encoder_broadcast, :
            ]
            encoder_hidden_states = new_encoder_hidden_states

        logits = self.trt_context.infer(
            {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states}
        )["hidden_states"]

        return Seq2SeqLMOutput(logits=torch.from_numpy(logits))


class T5Polygraphy(TRTInferenceCommand):
    def __init__(self):
        super().__init__(
            T5ModelTRTConfig,
            "Runs polygraphy results for T5 model. Only supports FP16 variants.",
            T5FHuggingFace,
        )
        self.t5_trt_decoder = None
        self.t5_trt_encoder = None

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_trt_engine: bool = False,
        keep_onnx_model: bool = False,
        keep_torch_model: bool = False,
    ) -> None:
        # Deactivates context
        self.t5_trt_encoder.release()
        self.t5_trt_decoder.release()

        if not keep_trt_engine:
            self.t5_trt_encoder_engine.cleanup()
            self.t5_trt_decoder_engine.cleanup()

        self.frameworks_cmd.cleanup(workspace, keep_onnx_model, keep_torch_model)

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Dict[str, NetworkModel],
        inference_input: str,
        timing_profile: TimingProfile,
    ) -> NetworkResult:

        tokenizer = T5Tokenizer.from_pretrained(metadata.variant)
        input_ids = tokenizer(inference_input, return_tensors="pt").input_ids
        encoder_last_hidden_state, encoder_e2e_median_time = encoder_inference(
            self.t5_trt_encoder, input_ids, timing_profile, use_cuda=False
        )
        _, decoder_e2e_median_time = decoder_inference(
            self.t5_trt_decoder,
            input_ids,
            encoder_last_hidden_state,
            timing_profile,
            use_cuda=False,
        )
        decoder_output_greedy, full_e2e_median_runtime = full_inference_greedy(
            self.t5_trt_encoder,
            self.t5_trt_decoder,
            input_ids,
            tokenizer,
            timing_profile,
            max_length=40,
            use_cuda=False,
        )

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.convert_ids_to_tokens(
            decoder_output_greedy.tolist()[0]
        )[1:-1]
        remove_underscore = "".join(
            [s.replace("\u2581", " ") for s in semantic_outputs]
        )

        return NetworkResult(
            input=inference_input,
            output_tensor=encoder_last_hidden_state,
            semantic_output=remove_underscore.strip(),
            median_runtime=[
                NetworkRuntime(
                    name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    runtime=decoder_e2e_median_time,
                ),
                NetworkRuntime(
                    name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                    runtime=encoder_e2e_median_time,
                ),
                NetworkRuntime(
                    name=T5ModelTRTConfig.NETWORK_FULL_NAME,
                    runtime=full_e2e_median_runtime,
                ),
            ],
            models=NetworkModels(
                torch=None,
                onnx=list(onnx_fpaths.values()),
                trt=[
                    NetworkModel(
                        name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                        fpath=self.t5_trt_decoder_engine.fpath,
                    ),
                    NetworkModel(
                        name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                        fpath=self.t5_trt_encoder_engine.fpath,
                    ),
                ],
            ),
        )

    def run_trt(
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
                T5ModelTRTConfig.NETWORK_SEGMENTS
            ), "There should only be {} exported ONNX segments in T5 model.".format(
                len(T5ModelTRTConfig.NETWORK_SEGMENTS)
            )

            hash_onnx_fpath = {v.name: v for v in onnx_fpaths}

            decoder_onnx_fpath = hash_onnx_fpath[
                T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
            ].fpath
            encoder_onnx_fpath = hash_onnx_fpath[
                T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME
            ].fpath

            self.t5_trt_encoder_engine = T5EncoderONNXFile(
                encoder_onnx_fpath, metadata
            ).as_trt_engine(encoder_onnx_fpath + ".engine")
            self.t5_trt_decoder_engine = T5DecoderONNXFile(
                decoder_onnx_fpath, metadata
            ).as_trt_engine(decoder_onnx_fpath + ".engine")
            tfm_config = T5Config(
                use_cache=metadata.other.kv_cache,
                num_layers=T5ModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant],
            )
            self.t5_trt_encoder = T5TRTEncoder(
                self.t5_trt_encoder_engine.fpath, metadata, tfm_config
            )
            self.t5_trt_decoder = T5TRTDecoder(
                self.t5_trt_decoder_engine.fpath, metadata, tfm_config
            )

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
        super().add_args(parser)
        polygraphy_group = parser.add_argument_group("polygraphy models")
        polygraphy_group.add_argument(
            "--onnx-decoder-fpath",
            default=None,
            help="Path to ONNX decoder. If None is supplied, scripts will generate them from HuggingFace.",
        )
        polygraphy_group.add_argument(
            "--onnx-encoder-fpath",
            default=None,
            help="Path to ONNX encoder. If None is supplied, scripts will generate them from HuggingFace.",
        )

    def args_to_network_models(self, args) -> List[NetworkModel]:
        # Check if both flags are given otherwise error out
        decoder_fpath_check = args.onnx_decoder_fpath is None
        encoder_fpath_check = args.onnx_encoder_fpath is None

        network_models = None
        if decoder_fpath_check and encoder_fpath_check:
            network_models = tuple()
        elif decoder_fpath_check or encoder_fpath_check:
            raise self._parser.error(
                "Both --onnx-decoder-fpath and --onnx-encoder-fpath must be given. Otherwise neither should be provided for script to download them."
            )
        else:
            onnx_decoder = NetworkModel(
                name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=args.onnx_decoder_fpath,
            )
            onnx_encoder = NetworkModel(
                name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                fpath=args.onnx_encoder_fpath,
            )
            network_models = (onnx_decoder, onnx_encoder)

        return network_models

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        frameworks_parsed_metadata = self.frameworks_cmd.args_to_network_metadata(args)

        return NetworkMetadata(
            variant=frameworks_parsed_metadata.variant,
            precision=Precision(fp16=args.fp16),
            other=frameworks_parsed_metadata.other,
        )


RUN_CMD = T5Polygraphy()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
