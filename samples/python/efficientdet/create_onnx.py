#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import argparse
import logging

import tensorflow as tf
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from onnx import shape_inference
from tf2onnx import tfonnx, optimizer, tf_loader

import onnx_utils

logging.basicConfig(level=logging.INFO)
logging.getLogger("EfficientDetHelper").setLevel(logging.INFO)
log = logging.getLogger("EfficientDetHelper")


class EfficientDetGraphSurgeon:
    def __init__(self, saved_model_path):
        """
        Constructor of the EfficientDet Graph Surgeon object, to do the conversion of an EfficientDet TF saved model
        to an ONNX-TensorRT parsable model.
        :param saved_model_path: The path pointing to the TensorFlow saved model to load.
        """
        saved_model_path = os.path.realpath(saved_model_path)
        assert os.path.exists(saved_model_path)

        # Use tf2onnx to convert saved model to an initial ONNX graph.
        graph_def, inputs, outputs = tf_loader.from_saved_model(saved_model_path, None, None, "serve",
                                                                ["serving_default"])
        log.info("Loaded saved model from {}".format(saved_model_path))
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name="")
        with tf_loader.tf_session(graph=tf_graph):
            onnx_graph = tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs, opset=11)
        onnx_model = optimizer.optimize_graph(onnx_graph).make_model("Converted from {}".format(saved_model_path))
        self.graph = gs.import_onnx(onnx_model)
        assert self.graph
        log.info("ONNX graph created successfully")

        # Try to auto-detect by finding if nodes match a specific name pattern expected for either of the APIs.
        self.api = None
        if len([node for node in self.graph.nodes if "class_net/" in node.name]) > 0:
            self.api = "AutoML"
        elif len([node for node in self.graph.nodes if "/WeightSharedConvolutionalClassHead/" in node.name]) > 0:
            self.api = "TFOD"
        assert self.api
        log.info("Graph was detected as {}".format(self.api))

        self.batch_size = None

    def infer(self):
        """
        Run shape inference on the ONNX graph to determine tensor shapes.
        """
        self.graph.cleanup().toposort()
        for node in self.graph.nodes:
            # Some tensor shapes get reset to dims of 0 which throws off the shape inferer, so better reset all tensors
            for o in node.outputs:
                o.shape = None
        model = shape_inference.infer_shapes(gs.export_onnx(self.graph))
        self.graph = gs.import_onnx(model)

    def save(self, output_path):
        """
        Save the ONNX model to the given location.
        :param output_path: Path pointing to the location where to write out the updated ONNX model.
        """
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        output_path = os.path.realpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save(model, output_path)
        log.info("Saved ONNX model to {}".format(output_path))

    def update_preprocessor(self, input_shape):
        """
        Remove all the pre-processing nodes in the ONNX graph and leave only the image normalization essentials.
        :param input_shape: The input tensor shape to use for the ONNX graph.
        """
        # Update the input and output tensors shape
        input_shape = input_shape.split(",")
        assert len(input_shape) == 4
        for i in range(len(input_shape)):
            input_shape[i] = int(input_shape[i])
            assert input_shape[i] >= 1
        input_format = None
        if input_shape[1] == 3:
            input_format = "NCHW"
        if input_shape[3] == 3:
            input_format = "NHWC"
        assert input_format in ["NCHW", "NHWC"]
        self.batch_size = input_shape[0]
        self.graph.inputs[0].shape = input_shape
        self.graph.inputs[0].dtype = np.float32
        log.info("ONNX graph input shape: {} [{} format detected]".format(self.graph.inputs[0].shape, input_format))

        # Find the initial nodes of the graph, whatever the input is first connected to, and disconnect them
        for node in [node for node in self.graph.nodes if self.graph.inputs[0] in node.inputs]:
            node.inputs.clear()

        # Convert to NCHW format if needed
        input_tensor = self.graph.inputs[0]
        if input_format == "NHWC":
            input_tensor = self.graph.transpose("preprocessor/transpose", input_tensor, [0, 3, 1, 2])

        # RGB Normalizers. The per-channel values are given with shape [1, 3, 1, 1] for proper NCHW shape broadcasting
        scale_val = 1 / np.asarray([255], dtype=np.float32)
        mean_val = -1 * np.expand_dims(np.asarray([0.485, 0.456, 0.406], dtype=np.float32), axis=(0, 2, 3))
        stddev_val = 1 / np.expand_dims(np.asarray([0.229, 0.224, 0.225], dtype=np.float32), axis=(0, 2, 3))
        # y = (x * scale + mean) * stddev   -->   y = x * scale * stddev + mean * stddev
        scale_out = self.graph.elt_const("Mul", "preprocessor/scale", input_tensor, scale_val * stddev_val)
        mean_out = self.graph.elt_const("Add", "preprocessor/mean", scale_out, mean_val * stddev_val)

        # Find the first stem conv node of the graph, and connect the normalizer directly to it
        stem_name = None
        if self.api == "AutoML":
            stem_name = "/stem/"
        if self.api == "TFOD":
            stem_name = "/stem_conv2d/"
        stem = [node for node in self.graph.nodes if node.op == "Conv" and stem_name in node.name][0]
        log.debug("Found {} node '{}' as stem entry".format(stem.op, stem.name))
        stem.inputs[0] = mean_out[0]

    def update_resize(self):
        """
        Updates the graph to replace the nearest neighbor resize ops by ResizeNearest_TRT TensorRT plugin nodes.
        """

        # Make sure to have the most recent shape inference results
        self.infer()

        count = 0
        for i, node in enumerate([node for node in self.graph.nodes
                                  if node.op == "Resize" and node.attrs['mode'] == "nearest"]):
            # The scale factor should always be 2.0, but it's a bit safer to extract this from the ONNX graph, just in
            # case. This can be done by analyzing the node input and parameters.

            # The size of the input tensor (last dimension only)
            in_size = node.inputs[0].shape[-1]
            # The infer operation doesn't find the output tensor shape for the Resize node correctly, but it can
            # be found through the 'sizes' input of the Resize node, which is for the shape to resize to, with some
            # manipulation, we can get the expected size of the output tensor (last dimension only)
            out_size = node.i(3).inputs[1].values[-1]
            # The scale is then just the ratio of these two
            scale = out_size / in_size

            self.graph.plugin(
                op="ResizeNearest_TRT",
                name="resize_nearest_{}".format(i),
                inputs=[node.inputs[0]],
                outputs=node.outputs,
                attrs={
                    'plugin_version': "1",
                    'scale': scale,
                })
            node.outputs.clear()
            log.debug(
                "Replaced {} node '{}' with a ResizeNearest_TRT plugin node with scale {}".format(
                    node.op, node.name, scale))
            count += 1
        log.info("Swapped {} Resize nodes with ResizeNearest_TRT plugin nodes".format(count))

    def update_nms(self, threshold=None, efficient_nms_plugin=True):
        """
        Updates the graph to replace the NMS op by BatchedNMS_TRT TensorRT plugin node.
        :param threshold: Override the score threshold value. If set to None, use the value in the graph.
        :param efficient_nms_plugin: When True, use the newer EfficientNMS plugin, otherwise if the current TensorRT
        installation does not yet support it, set to False to use the older (and slower) BatchedNMS plugin.
        """

        def find_head_concat(name_scope):
            # This will find the concatenation node at the end of either Class Net or Box Net. These concatenation nodes
            # bring together prediction data for each of 5 scales.
            # The concatenated Class Net node will have shape [batch_size, num_anchors, num_classes],
            # and the concatenated Box Net node has the shape [batch_size, num_anchors, 4].
            # These concatenation nodes can be be found by searching for all Concat's and checking if the node two
            # steps above in the graph has a name that begins with either "box_net/..." or "class_net/...".
            # While here, make sure that the 5 reshape nodes that feed into the concat take into account the correct
            # batch size for their reshape operation.
            for node in [node for node in self.graph.nodes if node.op == "Transpose" and name_scope in node.name]:
                concat = self.graph.find_descendant_by_op(node, "Concat")
                assert concat and len(concat.inputs) == 5
                for i in range(5):
                    concat.i(i).inputs[1].values[0] = self.batch_size
                log.debug("Found {} node '{}' as the tip of {}".format(concat.op, concat.name, name_scope))
                return concat

        def reshape_boxes_tensor(input_tensor):
            # This will insert a node so the coordinate tensors have the exact shape that NMS expects.
            # The default box_net has shape [batch_size, number_boxes, 4], so this Unsqueeze will insert a "1" dimension
            # in the second axis, to become [batch_size, number_boxes, 1, 4].
            # The function returns the output tensor of the Unsqueeze operation.
            return self.graph.unsqueeze("nms/box_net_reshape", input_tensor, [2])[0]

        def find_decoder_concat():
            # This will find the concatenation node at the end of the Box Decoder, which appears immediately after
            # the Box Net. This box decoder is constructed by tf2onnx when converting the model, and as such, it follows
            # ONNX-like NMS coding. This means that even though EfficientDet uses "Center+Size" (y, x, h, w) coding for
            # the box predictions and anchors, this box decoder converts the coordinate coding to the alternate "Corner"
            # format (y_min, x_min, y_max, x_max).
            # The node we're looking for can be found by starting from the ONNX NMS node, and walk up the graph 4 steps.
            nms_node = self.graph.find_node_by_op("NonMaxSuppression")
            node = self.graph.find_ancestor_by_op(nms_node, "Concat")
            assert node and len(node.inputs) == 4
            log.debug("Found {} node '{}' as the tip of the box decoder".format(node.op, node.name))
            return node

        def extract_anchors_tensor(box_net):
            # This will find the anchor data. This is available (one per coordinate) hardcoded on the ONNX graph as
            # constants within the box decoder nodes. Each of these four constants have shape [batch_size, num_anchors],
            # so some numpy operations are used to expand the dims and concatenate them as needed. The final anchor
            # tensor shape then becomes [1, num_anchors, 1, 4]. Note that '1' is kept as the first dim, regardess of
            # batch size, as it's not necessary to replicate the anchors for all images in the batch.
            # These constants can be found by starting from the Box Net, finding the "Split" operation that splits the
            # box prediction data into each of the four coordinates, and for each of these, walking down two steps in
            # graph until either an Add or Mul node is found. The second input of these nodes will be the anchor data.
            split = self.graph.find_descendant_by_op(box_net, "Split")
            assert split and len(split.outputs) == 4

            def get_anchor_np(output_idx, op):
                node = self.graph.find_descendant_by_op(split.o(0, output_idx), op)
                assert node
                val = np.squeeze(node.inputs[1].values)
                if len(val.shape) > 1:
                    val = val[0]
                return np.expand_dims(val, axis=(0, 2))

            anchors_y = get_anchor_np(0, "Add")
            anchors_x = get_anchor_np(1, "Add")
            anchors_h = get_anchor_np(2, "Mul")
            anchors_w = get_anchor_np(3, "Mul")
            anchors = np.concatenate([anchors_y, anchors_x, anchors_h, anchors_w], axis=2)
            return gs.Constant(name="nms/anchors:0", values=anchors)

        head_names = []
        if self.api == "AutoML":
            head_names = ["class_net/", "box_net/"]
        if self.api == "TFOD":
            head_names = ["/WeightSharedConvolutionalClassHead/", "/WeightSharedConvolutionalBoxHead/"]

        class_net = find_head_concat(head_names[0])
        class_net_tensor = class_net.outputs[0]

        box_net = find_head_concat(head_names[1])
        box_net_tensor = box_net.outputs[0]

        # NMS Configuration
        nms_node = self.graph.find_node_by_op("NonMaxSuppression")
        num_detections = 3 * int(nms_node.inputs[2].values)
        iou_threshold = float(nms_node.inputs[3].values)
        score_threshold = float(nms_node.inputs[4].values) if threshold is None else threshold
        num_classes = class_net.i().inputs[1].values[-1]
        normalized = True if self.api == "TFOD" else False

        # NMS Inputs and Attributes
        # NMS expects these shapes for its input tensors:
        # box_net: [batch_size, number_boxes, 1, 4]
        # class_net: [batch_size, number_boxes, number_classes]
        # anchors: [1, number_boxes, 1, 4] (if used)
        nms_op = None
        nms_attrs = None
        nms_inputs = None
        if efficient_nms_plugin:
            # EfficientNMS TensorRT Plugin
            # Fusing the decoder will always be faster, so this is the default NMS method supported. In this case,
            # three inputs are given to the NMS TensorRT node:
            # - The box predictions (from the Box Net node found above)
            # - The class predictions (from the Class Net node found above)
            # - The default anchor coordinates (from the extracted anchor constants)
            # As the original tensors from EfficientDet will be used, the NMS code type is set to 1 (Center+Size),
            # because this is the internal box coding format used by the network.
            anchors_tensor = extract_anchors_tensor(box_net)
            nms_inputs = [box_net_tensor, class_net_tensor, anchors_tensor]
            nms_op = "EfficientNMS_TRT"
            nms_attrs = {
                'plugin_version': "1",
                'background_class': -1,
                'max_output_boxes': num_detections,
                'max_output_boxes_per_class': -1,
                'score_threshold': score_threshold,
                'iou_threshold': iou_threshold,
                'score_sigmoid': True,
                'box_coding': 1,
            }
            nms_output_classes_dtype = np.int32
        else:
            # BatchedNMS TensorRT Plugin
            # Alternatively, the ONNX box decoder can be used. This will be slower, as more element-wise and non-fused
            # operations will need to be performed by TensorRT. However, it's easier to implement, so it is shown here
            # for reference. In this case, only two inputs are given to the NMS TensorRT node:
            # - The box predictions (already decoded through the ONNX Box Decoder node)
            # - The class predictions (from the Class Net node found above, but also needs to pass through a sigmoid)
            # This time, the box predictions will have the coordinate coding from the ONNX box decoder, so the NMS code
            # type is set to 0 (Corner).
            box_decoder = find_decoder_concat()
            box_decoder_tensor = reshape_boxes_tensor(box_decoder.outputs[0])
            class_net_sigmoid_tensor = self.graph.sigmoid("nms/class_net_sigmoid", class_net_tensor)[0]
            nms_inputs = [box_decoder_tensor, class_net_sigmoid_tensor]
            nms_op = "BatchedNMSDynamic_TRT"
            nms_attrs = {
                'plugin_version': "1",
                'shareLocation': True,
                'backgroundLabelId': -1,
                'numClasses': num_classes,
                'topK': 1024,
                'keepTopK': num_detections,
                'scoreThreshold': score_threshold,
                'iouThreshold': iou_threshold,
                'isNormalized': normalized,
                'clipBoxes': False,
                'scoreBits': 10,  # Older TensorRT versions will not support this parameter. If so, simply remove it.
            }
            nms_output_classes_dtype = np.float32

        # NMS Outputs
        nms_output_num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=[self.batch_size, 1])
        nms_output_boxes = gs.Variable(name="detection_boxes", dtype=np.float32,
                                       shape=[self.batch_size, num_detections, 4])
        nms_output_scores = gs.Variable(name="detection_scores", dtype=np.float32,
                                        shape=[self.batch_size, num_detections])
        nms_output_classes = gs.Variable(name="detection_classes", dtype=nms_output_classes_dtype,
                                         shape=[self.batch_size, num_detections])
        nms_output_indices = gs.Variable(name="detection_indices", dtype=nms_output_classes_dtype,
                                         shape=[self.batch_size * num_detections, 3])

        nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]
        if efficient_nms_plugin:
            nms_outputs.append(nms_output_indices)

        self.graph.plugin(
            op=nms_op,
            name="nms/non_maximum_suppression",
            inputs=nms_inputs,
            outputs=nms_outputs,
            attrs=nms_attrs)

        self.graph.outputs.clear()
        self.graph.outputs = nms_outputs

    def add_debug_output(self, debug):
        """
        Updates the graph to add new outputs for a given set of tensor names. This is useful for debugging the values
        of a particular node's output tensor or to get intermediate layer values.
        :param debug: A list of tensor names that should be added to the graph outputs.
        """
        tensors = self.graph.tensors()
        for n, name in enumerate(debug):
            if name not in tensors:
                log.warning("Could not find tensor '{}'".format(name))
            debug_tensor = gs.Variable(name="debug:{}".format(n), dtype=tensors[name].dtype)
            debug_node = gs.Node(op="Identity", name="debug_{}".format(n), inputs=[tensors[name]],
                                 outputs=[debug_tensor])
            self.graph.nodes.append(debug_node)
            self.graph.outputs.append(debug_tensor)
            log.info("Adding debug output '{}' for graph tensor '{}'".format(debug_tensor.name, name))


def main(args):
    if args.verbose:
        log.setLevel(logging.DEBUG)
    effdet_gs = EfficientDetGraphSurgeon(args.saved_model)
    effdet_gs.update_preprocessor(args.input_shape)
    effdet_gs.infer()
    if args.legacy_plugins:
        effdet_gs.update_resize()
    effdet_gs.update_nms(args.nms_threshold, not args.legacy_plugins)
    if args.debug:
        effdet_gs.add_debug_output(args.debug)
    effdet_gs.save(args.onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--saved_model", help="The TensorFlow saved model directory to load")
    parser.add_argument("-o", "--onnx", help="The output ONNX model file to write")
    parser.add_argument("-i", "--input_shape", default="1,512,512,3",
                        help="Set the input shape of the graph, as comma-separated dimensions in NCHW or NHWC format, "
                             "default: 1,512,512,3")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-t", "--nms_threshold", type=float, help="Override the score threshold for the NMS operation")
    parser.add_argument("-d", "--debug", action='append', help="Add an extra output to debug a particular tensor, "
                                                               "this argument can be used multiple times")
    parser.add_argument("--legacy_plugins", action="store_true", help="Use legacy plugins for support on TensorRT "
                                                                      "version lower than 8.")
    args = parser.parse_args()
    if not all([args.saved_model, args.onnx]):
        parser.print_help()
        print("\nThese arguments are required: --saved_model and --onnx")
        sys.exit(1)
    main(args)
