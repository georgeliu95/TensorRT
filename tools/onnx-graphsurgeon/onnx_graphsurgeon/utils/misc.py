from onnx_graphsurgeon.logger.logger import G_LOGGER

from collections import OrderedDict
from typing import List


def default_value(value, default):
    return value if value is not None else default


def remove_from_end(container, obj):
    if obj not in container:
        return None

    idx = reversed(container).index(obj)
    return container.pop(idx)
