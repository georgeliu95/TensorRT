# NVIDIA TensorRT Sample "SampleYOLO"

More details about YOLO network can be found at https://pjreddie.com/darknet/yolov1/


## How to get YOLO caffe model

Please download the YOLO caffe model from https://drive.google.com/uc?export=download&confirm=T1H4&id=0Bzy9LxvTYIgKMXdqS29HWGNLdGM and place it at `<TensorRT Directory>/data/yolo/`

Source: https://github.com/cinastanbean/caffe-yolo-1

## How to generate INT8 calibration batches

Run `PrepareINT8CalibrationBatches.sh` from `<TensorRT Directory>/samples/sampleYOLO` directory to download the PASCAL VOC 2007 dataset and create batches for INT8 calibration. It select 500 random JPEG images from dataset and convert to PPM images. These 500 PPM images are used to generate INT8 calibration batches. Keep the batches in `<TensorRT Directory>/data/yolo/batches/`.

If you want to use a different dataset to generate INT8 batches, please use `<TensorRT Directory>/samples/sampleYOLO/batchPrepare.py` and keep the batch files in `<TensorRT Directory>/data/yolo/batches`.

