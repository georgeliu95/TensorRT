# NVIDIA TensorRT Sample "SampleYOLOv2"

The details about YOLOv2 network can be found at https://pjreddie.com/darknet/yolov2/.
Please refer to _YOLO9000: Better, Faster, Stronger_ paper for more details about the YOLO v2, YOLO9000 networks.

YOLOV2 and YOLO9000 networks has extra layers which are not part of caffe. To build caffe with extra layers please follow the below instructions

## Instructions to build caffe with extra layers

1. Extract `caffeLayers.tar.gz`
2. Copy `ip_yv2_region_layer.cpp`, `ip_yv2_reorg_layer.cpp` to `<caffe directory>/src/caffe/layers/`
3. Copy `ip_yv2_region_layer.hpp`, `ip_yv2_reorg_layer.hpp` to `<caffe directory>/include/caffe/layers/`
4. Add the below content at the end of `<caffe directory>/src/caffe/proto.caffe.proto`:
	```
	message IPYv2ReorgParameter {
	    optional uint32 stride = 1 [default = 2];
	}

	message IPYv2RegionParameter {
	    repeated float anchors = 1 [packed = true];
	    optional bool bias_match = 2 [default = true];
	    optional uint32 classes = 3 [default = 80];
	    optional uint32 coords = 4 [default = 4];
	    optional uint32 num = 5 [default = 5];
	    optional bool softmax = 6 [default = true];
	    optional float jitter = 7 [default = 0.3];
	    optional bool rescore = 8 [default = true];
	}
	```
5. Add the below content to `LayerParameter` structure in `<caffe directory>/src/caffe/proto.caffe.proto`:
	```
	optional IPYv2ReorgParameter ipyv2_reorg_param = 791901;
	optional IPYv2RegionParameter ipyv2_region_param = 791902;
	```

6. Re-build caffe

## How to get YOLO v2, YOLO9000 caffe model files

Set `PYTHONPATH` variable to `<CAFFE_DIR>/python`

Run `prepareCaffeModel.sh` to generate caffemodel files

## How to generate INT8 calibration batches

Run `prepareINT8CalibrationBatches.sh` to generate INT8 bacthes. It select 500 random JPEG images from dataset and convert to PPM images. These 500 PPM images are used to generate INT8 calibration batches. The batches for YOLOv2 and YOLO9000 are stored at `<TensorRT Directory>/data/yolov2/batchesV2/` and `<TensorRT Directory>/data/yolov2/batches9000/` respectively.

If you want to use a different dataset to generate INT8 batches, please use `batchPrepare.py` and keep the batch files in respective paths.

