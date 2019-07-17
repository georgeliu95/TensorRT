# **MaskRCNN in TensorRT**
This repository provides a MaskRCNN inference sample running in native C++ TensorRT. Multiple supporting plugins are also provided for running the Keras MaskRCNN model in TensorRT. Guidelines and sample scripts for converting the KERAS model to TensorRT model are also provided.

## Introduction

### Train the model
The training framework of MaskRCNN can be found at Matterport's
MaskRCNN repository. We have verified the  trained model **(backbone: ResNet101 + FPN, dataset: coco)** provided in the [v2.0 release](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0). However, it is possible to train your own model with specific backbone and datasets.

### Convert .H5 model to .UFF

Guidelines and sample scripts for conversion are provided in [converted](converted/README.md) directory.

### Load the model and running in TensorRT
For correct parsing and running the public repo's MaskRCNN model, following plugins are implemented:

- ResizeNearest_TRT: Nearest Neighbor interpolation for resizing features. This works for FPN(Feature Pyramid Network) module; 
- ProposalLayer_TRT: Generate the first stage's proposals based on anchors and RPN's (Region Proposal Network) outputs(scores, bbox_deltas);
- PyramidROIAlign_TRT: Crop and resize the feature of ROIs (first stage's proposals) from the corresponding feature layer;
- DetectionLayer_TRT: Refine the first stage's proposals to produce final detections;
- SpecialSlice_TRT: A workaround plugin to slice detection output [y1, x1, y2, x2, class_id, score] to [y1, x1, y2 , x2] for a data with more than one index dimensions (batch_idx, proposal_idx, detections(y1, x1, y2, x2)). 

After registering the above plugins, one can load the converted model and run inference via TensorRT.

## Prerequisites
```
keras >= 2.1.3
tensorflow-gpu >= 1.9.0
```

## Configure
To change the input shape, you could modify the following parameter in *mrcnn_config.h*:
```c++
static const nvinfer1::DimsCHW IMAGE_SHAPE{3, 1024, 1024};
```

## Usage
`sample_maskRCNN` provides a common inference pipeline sample for test images.

```
./sample_maskRCNN -d /path/to/data 
```

> NOTE: The sample image data can be downloaded from: 
> wget https://cdn.pixabay.com/photo/2016/07/09/22/22/cat-1506960_960_720.jpg
> wget https://cdn.pixabay.com/photo/2014/11/19/07/06/boeing-537006_1280.jpg