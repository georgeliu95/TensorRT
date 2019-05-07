# Reformat free I/O sample

The intention of the reformat free I/O feature is to remove artificial restrictions on network inputs and outputs imposed by TensorRT. Ideally, the user specifies input/output tensors format that are natively supported by TensorRT, removing the overhead of tensor reformats.


# What does this sample do

The sample gives a simple example of how to use the reformat free I/O API. In the network construction phase, users are able to set the allowed format
for the input and output layers respectively. This can be done via the API setAllowedFormats() of the NetworkTensor class.

Setting a format prevents added reformat layer; the input is expected to be in the format specified, and the output will be provided in the format specified.

Another thing is that INT8 I/O is also supported now. Notice that INT8 I/O is a valid format, with ranges [-128, 127] and dynamic.