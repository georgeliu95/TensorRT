# About This Sample
This sample demonstrates how to create an int8 calibrator, build and calibrate an engine for int8 mode,
and finally run inference in int8 mode.

During calibration, the calibrator retrieves a total of 1003 batches, with 100 images each. We have
simplified the process of reading and writing a calibration cache in Python, so that it is now possible
to easily cache calibration data to speed up engine builds (see `calibrator.py` for implementation details).

During inference, the sample loads a random batch from the calibrator, then performs inference on the
whole batch of 100 images.

# Installing Prerequisites
1. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the top-level of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the top-level of this sample.

# Running the Sample
1. Create a TensorRT inference engine, perform int8 calibration and run inference:
    ```
    python sample.py [-d DATA_DIR]
    ```
    The data directory needs to be specified only if TensorRT is not installed in the default location.
