from networks import Precision, NetworkMetadata, NNConfig
from collections import namedtuple
from itertools import product

# Limitation of namedtuples. You must declare namedtuples in module scope and not in classes.
# Otherwise pickle doesn't work.
# See: https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
T5Metadata = namedtuple("T5Metadata", ["kv_cache"])


class T5ModelTRTConfig(NNConfig):
    def __init__(self):
        main_variants = ["large", "base", "small"]
        precision_fp16 = [False]
        precision_int8 = [False]
        kv_caches = [False, True]

        variants = []
        for variant, fp16, int8, kv_cache in product(
            main_variants, precision_fp16, precision_int8, kv_caches
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
