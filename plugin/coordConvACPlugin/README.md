# coordConvACPlugin

**Table Of Contents**
- [coordConvACPlugin](#coordconvacplugin)
  - [Description](#description)
    - [Structure](#structure)
  - [Additional resources](#additional-resources)
  - [License](#license)
  - [Changelog](#changelog)
  - [Known issues](#known-issues)

## Description

The `coordConvACPlugin` is used for the CoordConv layer, which is generally used in deep learning models instead of standard Conv layers. 

The `coordConvACPlugin` implements the layer that adding this two channels to the input data with CUDA API and after that regular Convolution can be applied.

Each node with the op name `CoordConvAC` in `ONNX` graph will be mapped to that plugin. `Conv` node should follow after each `CoordConvAC` node into `ONNX` graph. 

Layer was introduced by Uber AI Labs at 2018. CoordConv layer improves Conv operation by adding a couple of channels with relative coordinates to the input data and after that doing regular Conv over new expanded matrix. 2 channels of relative coordinates allows Convolution to catch information about concrete place where this convolution were applied.

If input data for Conv layer is `X` with shape of `[N, C, H, W]`, where `N` is the batch size, `C` is the number of channels, `H` is the height, `W` is the width. Then in CoordConv layer for each `N`(image/matrix in batch) input data concatenates with 2 addictional channels with shapes `[1, C, H, 1]` at the end. First channel contains relative coordinates along the Y axis and the second channel contains coordinates along the X axis. As a result we are getting new input data with shapes `[N, C+2, H, W]` and applying regular Conv operation over new data.

Relative coordinates it's values in range `[-1; 1]` where `-1` - this is the values of the top row for 1st channel (Y axis) and values for the left column of 2nd channel (X axis). `1` - this is values for bottom row for 1st channel (Y axis) and values for the right column of 2nd channel (X axis). All other (middle) values of the matrices fill in by adding constant value that allow to came from -1 to 1.

Formula for counting constant step value is:

`STEP_VALUE_H = 2 / (H - 1)` - step value for the first channel

`STEP_VALUE_W = 2 / (W - 1)` - step value for the second channel

Thare are examples of 1st and 2nd channels for the input data with H=5 and W=5. STEP_VALUE_H = 0.5 and STEP_VALUE_W = 0.5

1st channel with Y relative coordinates

| | | | | | 
| ------------- |:-------------:| -----|-------------| -----:|
| -1	| -1	| -1	| -1	| -1 |
| -0.5 | 	-0.5 | 	-0.5 | 	-0.5 | 	-0.5 | 
| 0 | 	0 | 	0 | 	0 | 	0 | 
| 0.5 | 	0.5 | 	0.5 | 	0.5 | 	0.5 | 
| 1 | 	1 | 	1 | 	1 | 	1 | 

2nd channel with X relative coordinates

|     |     |     |     |     | 
| ------------- |:-------------:| -----|-------------| -----:|
| -1	| -0.5	| 0	| 0.5	| 1 |
|  -1 | 	-0.5 | 	0 | 	0.5 | 	1 | 
|  -1 | 	-0.5 | 	0 | 	0.5 | 	1 | 
|  -1 | 	-0.5 | 	0 | 	0.5 | 	1 | 
|  -1 | 	-0.5 | 	0 | 	0.5 | 	1 | 

This two matrices will be concatenated with input data by the formula `CONCAT([INPUT_DATA, 1ST_CHANNEL, 2ND_CHANNEL])` in channel dimension.

  
### Structure

This plugin takes one input and generates one output. It has a shape of `[N, C + 2, H, W]`, where `N` is the batch size, `C + 2` is the number of channels + 2 additional channels, `H` is the height, `W` is the width. 

## Additional resources

The following resources provide a deeper understanding of the `coordConvACPlugin` plugin:

**Networks**  
- Paper about Coord Conv layer ["An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"](https://arxiv.org/abs/1807.03247)    
- Blog post by Uber AI Labs about [CoordConv layer](https://eng.uber.com/coordconv/)
- Open=source implementation of the layer in PyTorch #1 [link](https://github.com/walsvid/CoordConv)
- Open-source implementation of the layer in PyTorch #2 [link](https://github.com/mkocabas/CoordConv-pytorch)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

April 2020
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
