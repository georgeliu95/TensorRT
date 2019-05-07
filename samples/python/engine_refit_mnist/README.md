# About This Sample
This sample demonstrates the engine refit functionality provided by TensorRT.
The model first trains an mnist model in PyTorch, then recreates the network in
TensorRT. In the first pass the weights for one of the conv layers (conv_1) is 
fed with dummy values resulting in an incorrect inference result. In the second 
pass we refit the engine with the trained weights for the conv_1 layer and run 
inference.

# Installing Prerequisites
1. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the top-level of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the top-level of this sample.

# Running the Sample
1. Create a TensorRT inference engine and run inference:
    ```
    python sample.py [-d DATA_DIR]
    ```
    The data directory needs to be specified only if TensorRT is not installed in the default location.
