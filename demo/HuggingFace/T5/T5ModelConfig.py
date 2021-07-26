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
T5Metadata = namedtuple("T5Metadata", ["kv_cache"])


class T5ModelTRTConfig(NNConfig):

    TARGET_MODELS = ["t5-small", "t5-base", "t5-large"]
    NUMBER_OF_LAYERS = {TARGET_MODELS[0]: 6, TARGET_MODELS[1]: 12, TARGET_MODELS[2]: 24}
    MAX_SEQUENCE_LENGTH = {TARGET_MODELS[0]: 512, TARGET_MODELS[1]: 768, TARGET_MODELS[2]: 1024}

    def __init__(self):
        precision_fp16 = [False]
        precision_int8 = [False]
        kv_caches = [False, True]

        variants = []
        for variant, fp16, int8, kv_cache in product(
            T5ModelTRTConfig.TARGET_MODELS, precision_fp16, precision_int8, kv_caches
        ):
            variants.append(
                NetworkMetadata(
                    variant=variant,
                    precision=Precision(fp16=fp16, int8=int8),
                    other=T5Metadata(kv_cache=kv_cache),
                )
            )

        super().__init__("T5", variants=variants)

    def get_python_requirements():
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.6.1")
        return base_requirements

    @staticmethod
    def get_input_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_inputs = Dims(
            OrderedDict(
                {
                    "input_ids": (Dims.BATCH, Dims.SEQUENCE),
                    "encoder_hidden_states": (Dims.BATCH, Dims.SEQUENCE, T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]),
                }
            )
        )

        encoder_inputs = Dims(OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)}))

        return {
            "decoder": decoder_inputs,
            "encoder": encoder_inputs
        }

    @staticmethod
    def get_output_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of output dimensions.

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_outputs = Dims(OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE)}))
        encoder_outputs = Dims(OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE, T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant])}))

        return {
            "decoder": decoder_outputs,
            "encoder": encoder_outputs
        }
