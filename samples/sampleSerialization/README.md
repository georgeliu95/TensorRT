# NVIDIA TensorRT Sample "sampleSerialization"

The sampleSerialization sample demonstrates how to:
- Parse a saved model in Caffe/ONNX/UFF format and serialize it
- Deserialize the TensorRT network and run inference on it

## Usage

This sample can be run as:

    ./sample_serialization [-h/--help] [-d/--datadir /path/to/data/dir/] [-f/--format {caffe, onnx, uff, trt}] [--caffe_prototxt /path/to/caffe/prototxt/file] [--caffe_weights /path/to/caffe/caffemodel/file] [--onnx_file /path/to/onnx/file] [--uff_file /path/to/uff/file] [--input_name name_of_input_tensor] [--output_name name_of_output_tensor] [--input_shape shape_of_input_tensor] [-o/--output /path/to/output/file] [--run_inference]

Example usage:

Caffe model:

    ./sample_serialization -f caffe --caffe_prototxt ResNet50_N2.prototxt --caffe_weights ResNet50_fp32.caffemodel --datadir $DIT/data/ResNet50/ -output_name prob -o serialized_ResNet50_caffe.tnb

ONNX model:

    ./sample_serialization -f onnx --onnx_file ResNet50.onnx --datadir $DIT/data/ResNet50/ -o serialized_ResNet50_onnx.tnb

UFF model:

    ./sample_serialization -f uff --uff_file tf2trt_resnet50.uff --datadir $DIT/data/ResNet50/ --input_name input --input_shape 1,3,224,224 -o serialized_ResNet50_uff.tnb

Serialized TensorRT network:

    ./sample_serialization -f trt --trt_file serialized_ResNet50_caffe.tnb --datadir $DIT/data/ResNet50/ --run_inference

The input format can be set by setting `--format` argument to one of: `caffe`, `onnx`, `uff`, `trt`.
By default it is set to `caffe`.

SampleSerialization reads files to build the network:

Caffe:
* `--caffe_prototxt model.prototxt` - The prototxt file that contains the network design.
* `--caffe_weights model.caffemodel` - The model file which contains the trained weights for the network.
* `--output_name name_of_output_tensor` - The name of output tensor. This can be provided multiple times to set multiple output tensors.

ONNX:
* `--onnx_file model.onnx` - The ONNX file that contains network and weights.

UFF:
* `--uff_file model.uff` - The UFF file that contains network and weights.
* `--input_name name_of_input_tensor` - The name of the input tensor. This can be provided multiple times to add multiple input tensors.
* `--input_shape shape_of_input_tensor` - The shape of input tensor, specified as a comma separated list of numbers (NO SPACES). This option must be provided the same number of times as `--input_name` option. This i'th `--input_shape` argument corresponds to i'th `--input_name` argument.

TensorRT:
* `--trt_file model.tnb` - The file with serialized TensorRT network.


By default, the sample expects these files to be in `data/ResNet50/`.
The default directory can be changed by supplying the path as
`--datadir=/new/path/` as a command line argument.
