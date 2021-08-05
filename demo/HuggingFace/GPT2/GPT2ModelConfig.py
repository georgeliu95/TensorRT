# std
from collections import namedtuple, OrderedDict
from itertools import product
from typing import Dict

# TRT-HuggingFace
from models import Dims
from networks import Precision, NetworkMetadata, NNConfig

# Limitation of namedtuples. You must declare namedtuples in module scope and not in classes.
# Otherwise pickle doesn't work.
# See: https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
GPT2Metadata = namedtuple("GPT2Metadata", ["kv_cache"])


class GPT2ModelTRTConfig(NNConfig):
    VOCAB_SIZE = 50257 # Vocabulary size of the GPT-2 model
    TARGET_MODELS = ["gpt2", "gpt2-large"]
    NETWORK_DECODER_SEGMENT_NAME = "gpt2_decoder"
    NETWORK_SEGMENTS = [NETWORK_DECODER_SEGMENT_NAME]
    MAX_SEQUENCE_LENGTH = {
        TARGET_MODELS[0] : 64,
    }
    
    def __init__(self):
        precision_fp16 = [False, True]
        kv_caches = [False]
        variants = []
        for variant, fp16, kv_cache in product(
            GPT2ModelTRTConfig.TARGET_MODELS, precision_fp16, kv_caches
        ):
            variants.append(
                NetworkMetadata(
                    variant=variant,
                    precision=Precision(fp16=fp16),
                    other=GPT2Metadata(kv_cache=kv_cache),
                )
            )

        super().__init__("GPT2", variants=variants)

    def get_python_requirements(self):
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.6.1")
        return base_requirements

    @staticmethod
    def get_input_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Returns:
            (Dict[str, Dims]): {"decoder": Dims}
        """
        return {
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: Dims(OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)})),
        }

    @staticmethod
    def get_output_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of output dimensions.

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        return {
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: Dims(OrderedDict({"logits": (Dims.BATCH, Dims.SEQUENCE, GPT2ModelTRTConfig.VOCAB_SIZE)})),
        }
