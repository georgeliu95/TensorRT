# NVIDIA TensorRT Internal Sample "sampleAV"

The sampleAV sample runs FP32, FP16 and INT8 inference on Automotive Networks.

# Network supported:
1. drivenet_GridboxModel
2. mapnet_v1_8class
3. DriveNet
4. LaneNet
5. OpenRoadNet

# Prepare data for the sample
1. Map calibration and inference batch data to DIT/data/int8_samples/AV/
    make install_int8_data

# Usage

1. This sample can be run as:
    Usage:
    1. Print Help Information: 
        ./sample_av [-h or --help]
    2. Run:
        ./sample_av [--network_name=network_name] [--num_calib_batches=N] [--calib_batch_size=N] [--num_infer_batches=N] [--infer_batch_size=N] [--input_path=/path/to/input/data/dir] [--output_path=/path/to/output/data/dir] [--useDLACore=<int>] [--int8_tolerance=<float>] [--fp16_tolerance=<float>]

By default, the sample expects these files to be in `data/int8_samples/AV/` or `data/samples/AV` or `data/AV`. Default input directories can be changed by adding --input_path=/path/to/input/data/dir.
