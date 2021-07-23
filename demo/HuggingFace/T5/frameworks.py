# std
import os
import sys
import argparse

from typing import List


if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# huggingface
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)

# torch
import torch

# helpers
from interface import FrameworkCommand
from networks import (
    NetworkResult,
    NetworkMetadata,
    Precision,
    NetworkModels,
    NNFolderWorkspace,
    TimingProfile,
)
from T5.export import T5EncoderTorchFile, T5DecoderTorchFile
from T5.T5ModelConfig import T5ModelTRTConfig
from utils import confirm_folder_delete, measure_frameworks_inference_speed


TARGET_MODELS = ["t5-small", "t5-base", "t5-large"]
NUMBER_OF_LAYERS = {TARGET_MODELS[0]: 6, TARGET_MODELS[1]: 12, TARGET_MODELS[2]: 24}
NETWORK_INPUT_DEFAULT = [
    "premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity.",
    "translate English to German: That is good.",
    "cola sentence: All your base are belong to us.",
    "premise: If I fall sleep then I am going to wake up in 8 hours. hypothesis: I fell asleep but did not wake up in 8 hours.",
]


class T5FHuggingFace(FrameworkCommand):
    def __init__(self):
        super().__init__(
            T5ModelTRTConfig, description="Runs framework results for T5 model."
        )

        # Default inference input used during inference stage
        self.inference_input = NETWORK_INPUT_DEFAULT

        self.onnx_t5_encoder = None
        self.onnx_t5_decoder = None
        self.torch_t5_dir = None

    def generate_and_download_framework(
        self, metadata: NetworkMetadata, workspace: NNFolderWorkspace
    ) -> None:

        model_fullvariant = "t5-{}".format(metadata.variant)
        if model_fullvariant not in TARGET_MODELS:
            raise RuntimeError(
                "{} is not a supported variant for T5".format(metadata.variant)
            )

        cache_variant = False
        if metadata.other.kv_cache:
            cache_variant = True

        trt_t5_config = self.config
        metadata_serialized = trt_t5_config.get_metadata_string(metadata)
        workspace_dir = workspace.get_path()

        pytorch_model_dir = os.path.join(workspace_dir, metadata_serialized)
        # We keep track of the generated torch location for cleanup later
        self.torch_t5_dir = pytorch_model_dir

        model = None
        tfm_config = T5Config(
            use_cache=cache_variant, num_layers=NUMBER_OF_LAYERS[model_fullvariant]
        )
        if not os.path.exists(pytorch_model_dir):
            # Generate the pre-trained weights
            model = T5ForConditionalGeneration(tfm_config).from_pretrained(
                model_fullvariant
            )
            model.save_pretrained(pytorch_model_dir)
            print("Pytorch Model saved to {}".format(pytorch_model_dir))
        else:
            print(
                "Frameworks file already exists, skipping generation and loading from file instead."
            )
            model = T5ForConditionalGeneration(tfm_config).from_pretrained(
                pytorch_model_dir
            )

        # These ONNX models can be converted using special encoder and decoder classes.
        root_onnx_model_name = "{}.onnx".format(metadata_serialized)
        root_onnx_model_fpath = os.path.join(
            os.getcwd(), workspace_dir, root_onnx_model_name
        )
        encoder_onnx_model_fpath = root_onnx_model_fpath + "-encoder.onnx"
        decoder_onnx_model_fpath = root_onnx_model_fpath + "-decoder-with-lm-head.onnx"

        t5_encoder = T5EncoderTorchFile(model)
        t5_decoder = T5DecoderTorchFile(model)
        self.onnx_t5_encoder = t5_encoder.as_onnx_model(
            encoder_onnx_model_fpath, force_overwrite=False
        )
        self.onnx_t5_decoder = t5_decoder.as_onnx_model(
            decoder_onnx_model_fpath, force_overwrite=False
        )

        onnx_fpaths = (self.onnx_t5_decoder.fpath, self.onnx_t5_encoder.fpath)
        torch_fpaths = (pytorch_model_dir,)

        return NetworkModels(torch=torch_fpaths, onnx=onnx_fpaths)

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        save_onnx_model: bool = True,
        save_pytorch_model: bool = False,
    ) -> None:
        """
        Cleans up the working directory and leaves models if available.
        Returns:
            None
        """
        # Clean-up generated files
        if not save_onnx_model:
            self.onnx_t5_decoder.cleanup()
            self.onnx_t5_encoder.cleanup()

        if not save_pytorch_model:
            # Using rmtree can be dangerous, have user confirm before deleting.
            confirm_folder_delete(
                self.torch_t5_dir,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

        if not save_pytorch_model and not save_onnx_model:
            workspace.cleanup(force_remove=False)

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        network_fpaths: NetworkModels,
        inference_input: str,
        timing_profile: TimingProfile,
    ) -> NetworkResult:

        # Execute some tests
        full_variant_name = "t5-{}".format(metadata.variant)
        tokenizer = T5Tokenizer.from_pretrained(full_variant_name)
        input_ids = tokenizer(inference_input, return_tensors="pt").input_ids

        # By default, huggingface model structure is one giant file.
        t5_torch_fpath = network_fpaths.torch[0]
        config = T5Config(
            use_cache=metadata.other.kv_cache,
            num_layers=NUMBER_OF_LAYERS[full_variant_name],
        )
        t5_model = T5ForConditionalGeneration(config).from_pretrained(t5_torch_fpath)

        t5_torch_encoder = T5EncoderTorchFile.TorchModule(t5_model.encoder)
        t5_torch_decoder = T5DecoderTorchFile.TorchModule(
            t5_model.decoder, t5_model.lm_head, t5_model.config
        )
        encoder_stmt = lambda: t5_torch_encoder(input_ids=input_ids)

        fastest_time = measure_frameworks_inference_speed(
            encoder_stmt,
            number=timing_profile.number,
            iterations=timing_profile.iterations,
        )

        # Finally get the result
        encoder_last_hidden_state = t5_torch_encoder(input_ids=input_ids)

        # Preprocess for semantic output
        # Set beams to one because of greedy algorithm
        num_beams = 1
        decoder_input_ids = torch.full(
            (num_beams, 1), tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        )
        encoder_last_hidden_state = torch.cat([encoder_last_hidden_state] * num_beams)
        decoder_output_greedy = t5_torch_decoder.greedy_search(
            input_ids=decoder_input_ids, encoder_hidden_states=encoder_last_hidden_state
        ).detach()

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.convert_ids_to_tokens(
            decoder_output_greedy.tolist()[0]
        )[1:-1]
        remove_underscore = "".join(
            [s.replace("\u2581", " ") for s in semantic_outputs]
        )

        return NetworkResult(
            output_tensor=encoder_last_hidden_state,
            semantic_output=remove_underscore.strip(),
            median_runtime=fastest_time,
            models=network_fpaths,
        )

    def run_framework(
        self,
        metadata: NetworkMetadata,
        network_input: List[str],
        working_directory: str,
        save_onnx_model: bool,
        save_pytorch_model: bool,
        timing_profile: TimingProfile,
    ) -> List[NetworkResult]:
        """
        Main entry point of our function which compiles and generates our model data.
        """

        results = []
        workspace = NNFolderWorkspace(
            self.config.network_name, metadata, working_directory
        )
        try:
            network_fpaths = self.generate_and_download_framework(metadata, workspace)
            for ninput in network_input:
                results.append(
                    self.execute_inference(
                        metadata, network_fpaths, ninput, timing_profile
                    )
                )
        finally:
            self.cleanup(workspace, save_onnx_model, save_pytorch_model)

        return results

    def add_args(self, parser) -> NetworkMetadata:
        super().add_args(parser)
        parser.add_argument(
            "--variant",
            help="T5 variant to generate",
            choices=TARGET_MODELS,
            required=True,
        )
        parser.add_argument(
            "--enable-kv-cache",
            help="T5 enable KV cache",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--save-onnx-model",
            help="Save the onnx model.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--save-torch-model",
            help="Save the torch model in model directory.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--working-dir",
            help="Location of where to save the model if --save-* is enabled.",
            required=True,
        )

    def args_to_network_metadata(self, args: argparse.Namespace) -> NetworkMetadata:
        return NetworkMetadata(
            variant=args.variant.lstrip("t5-"),
            precision=Precision(fp16=False, int8=False),
            other=self.config.MetadataClass(kv_cache=args.enable_kv_cache),
        )


# Entry point
RUN_CMD = T5FHuggingFace()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
