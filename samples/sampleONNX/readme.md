sampleONNX

This sample demonstrates how to runs TensorRT on various ONNX models and use various precision.

Building of this sample is different compared to most of other samples because it depends on the open source code.

To build the sampleONNX

1) mkdir build
2) cd build
3) Setup env variables:
   a) export TRT_ROOT=<>
   b) export EXT_PATH=<path to ...DIT/externals
   
4)  ../setup_SampleOnnxCmd.sh -z /usr/bin/cmake -o $TRT_ROOT/build/cuda-10.0/7.2/x86_64 -p x86_64 --cuda_root /usr/local/cuda-10.0/targets/x86_64-linux --cudnn_root $EXT_PATH/cudnn/x86_64/7.2/cuda-10.0 -m ..

5) make
6) make install


This sample accepts multiple options on how to run.
Please type ./sampleONNX without any arguments to get the usage.


