from networks import Precision, NetworkMetadata, NNConfig
from collections import namedtuple

# Limitation of namedtuples. You must declare namedtuples in module scope and not in classes.
# Otherwise pickle doesn't work.
# See: https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
T5Metadata = namedtuple("T5Metadata", ["kv_cache"])


class T5ModelTRTConfig(NNConfig):
    def __init__(self):
        super().__init__(
            "T5",
            variants=[
                NetworkMetadata(
                    variant="small",
                    precision=Precision(fp16=False, int8=False),
                    other=T5Metadata(kv_cache=False),
                )
            ],
        )

    def get_python_requirements():
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.6.1")
        return base_requirements
