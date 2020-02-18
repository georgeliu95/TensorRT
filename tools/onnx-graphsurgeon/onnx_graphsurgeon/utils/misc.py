from onnx_graphsurgeon.logger.logger import G_LOGGER

from collections import OrderedDict
from typing import List, Sequence


def default_value(value, default):
    return value if value is not None else default


# Special type of list that synchronizes contents with another list.
# Concrete example: Assume some node, n, contains an input tensor, t. If we remove t from n.inputs,
# we also need to remove n from t.outputs. To avoid having to do this manually, we use SynchronizedList,
# which takes an attribute name as a parameter, and then synchronizes to that attribute of each of its elements.
# So, in the example above, we can make n.inputs a synchronized list whose field_name is set to "outputs".
# See test_ir.TestNodeIO for functional tests
class SynchronizedList(list):
    def __init__(self, parent_obj, field_name, initial):
        self.parent_obj = parent_obj
        self.field_name = field_name
        self.extend(initial)


    def _add_to_elem(self, elem):
        # Explicitly avoid SynchronizedList overrides to prevent infinite recursion
        list.append(getattr(elem, self.field_name), self.parent_obj)


    def _remove_from_elem(self, elem):
        # Explicitly avoid SynchronizedList overrides to prevent infinite recursion
        list.remove(getattr(elem, self.field_name), self.parent_obj)


    def __delitem__(self, index):
        self._remove_from_elem(self[index])
        super().__delitem__(index)


    def __setitem__(self, index, elem):
        self._remove_from_elem(self[index])
        super().__setitem__(index, elem)
        self._add_to_elem(elem)


    def append(self, x):
        super().append(x)
        self._add_to_elem(x)


    def extend(self, iterable: Sequence[object]):
        super().extend(iterable)
        for elem in iterable:
            self._add_to_elem(elem)

    def insert(self, i, x):
        super().insert(i, x)
        self._add_to_elem(x)


    def remove(self, x):
        super().remove(x)
        self._remove_from_elem(x)


    def pop(self, i=-1):
        elem = super().pop(i)
        self._remove_from_elem(elem)
        return elem


    def clear(self):
        for elem in self:
            self._remove_from_elem(elem)
        super().clear()


    def __add__(self, other_list: List[object]):
        new_list = SynchronizedList(self.parent_obj, self.field_name, self)
        new_list += other_list
        return new_list


    def __iadd__(self, other_list: List[object]):
        self.extend(other_list)
        return self
