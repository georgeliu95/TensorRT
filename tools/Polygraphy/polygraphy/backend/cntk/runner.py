# EXPERIMENTAL
from polygraphy.logger.logger import G_LOGGER
from polygraphy.backend.base import BaseRunner

from collections import OrderedDict
import time

import cntk


class CNTKRunner(BaseRunner):
    def __init__(self, model, name=None):
        super().__init__(name=name, prefix="cntk-runner")
        self.model = model


    def activate_impl(self):
        self.cntk_model = cntk.Function.load(self.model)

        self.inputs = OrderedDict()
        for inp in self.cntk_model.arguments:
            self.inputs[inp] = inp.shape


    def infer(self, feed_dict):
        start = time.time()
        inference_outputs = self.cntk_model.eval(feed_dict)
        end = time.time()

        out_dict = OrderedDict()
        for out_node, out in zip(self.cntk_model.outputs, inference_outputs):
            out_dict[out_node.name] = out

        self.inference_time = end - start
        return out_dict
