# TensorRT FP8 Inference for NeMo models
This repository demonstrates TensorRT inference with NeMo Megatron models in FP8/FP16/BF16 precision.

Currently, this repository supports [NeMo GPT](https://huggingface.co/nvidia/nemo-megatron-gpt-5B) models only.

# Environment Setup
It's recommended to run inside a container to avoid conflicts when installing dependencies. Tested with TensorRT container [`tensorrt:23.04-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags). A GPU with compute capability 9.0 or above is required to run the demo.

```
source install.sh
```

> The script will install required dependencies and it can take more than 30 minutes.

**Please note that the [HuggingFace demo directory](demo/HuggingFace) needs to be visible when running this demo, so utility functions can be correctly imported.**

# File Structure
This demo follows simliar structure and command-line interface as in [HuggingFace demo](demo/HuggingFace).
```
.
├── GPT3                   # GPT3 directory
│   ├── GPT3ModelConfig.py # model configuration and variant-specific parameters
│   ├── frameworks.py      # NeMo PyTorch inference script
│   ├── onnxrt.py          # OnnxRT inference script
│   ├── trt.py             # TensorRT inference script
│   ├── decoding.py        # main inference logic for all runtimes
│   ├── export_utils.py    # export functions for NeMo model -> ONNX model -> TRT engine
│   └── ...                # files with utility functions for export and inference
├── interface.py # definitions of setup functions
└── run.py       # main entry script
```

# How to run inference
The `run` action will run end-to-end inference on sentences specified in [megatron_gpt_demo.yaml](demo/NeMo/GPT3/megatron_gpt_demo.yaml).
```
python3 run.py run GPT3 [frameworks | trt] --variant gpt-5b --working-dir $(pwd)/temp --fp8 --fp16
```

Expected output for the second sentence:
```
Batch 1: {'sentences': ['Tell me an interesting fact about TensorRT.\n\nTensorRT is a library for running machine learning algorithms on TensorFlow.\n\nTensorRT is a library for running machine learning'],
          'tokens': [['<|endoftext|>', 'Tell', ' me', ' an', ' interesting', ' fact', ' about', ' T', 'ensor', 'RT', '.', '\n', '\n', 'T', 'ensor', 'RT', ' is', ' a', ' library', ' for', ' running', ' machine', ' learning', ' algorithms', ' on', ' T', 'ensor', 'Flow', '.', '\n', '\n', 'T', 'ensor', 'RT', ' is', ' a', ' library', ' for', ' running', ' machine', ' learning']],
          'logprob': tensor([[-8.9306e+00, -1.9799e+00, -7.5495e+00, -2.1066e+00, -7.3745e-01,
         -4.1447e-01, -7.8721e+00, -6.5872e+00, -6.7521e+00, -1.3699e+00,
         -5.6391e-01, -1.6518e-01, -2.2900e+00, -2.2496e-02, -1.0490e-01,
         -4.6875e-01, -5.9859e-01, -2.7474e+00, -9.9975e-01, -2.6057e+00,
         -1.9231e+00, -8.5407e-02, -9.6093e-01, -5.4157e-01, -2.0662e+00,
         -6.8296e-02, -5.9870e-02, -1.0722e+00, -1.3832e+00, -9.8978e-02,
         -1.4420e+00, -1.2904e-02, -6.5171e-01, -8.1253e-01, -1.6091e+00,
         -1.1594e+00, -1.3299e-01, -9.7715e-02, -2.2241e-01, -6.7608e-03]],
       device='cuda:0'),
          'full_logprob': None,
          'token_ids': [[50256, 24446, 502, 281, 3499, 1109, 546, 309, 22854, 14181, 13, 198, 198, 51, 22854, 14181, 318, 257, 5888, 329, 2491, 4572, 4673, 16113, 319, 309, 22854, 37535, 13, 198, 198, 51, 22854, 14181, 318, 257, 5888, 329, 2491, 4572, 4673]],
          'offsets': [[0, 0, 4, 7, 10, 22, 27, 33, 35, 40, 42, 43, 44, 45, 46, 51, 53, 56, 58, 66, 70, 78, 86, 95, 106, 109, 111, 116, 120, 121, 122, 123, 124, 129, 131, 134, 136, 144, 148, 156, 164]]}
```

# How to run with various configurations
- FP8, FP16, and BF16 precisions are supported, and they can be set through `--fp8`, `--fp16`, and `--bf16` respectively. Currently, the script has constraints on how precisions are specified, and supported combinations are:
  1. Pure FP16: `--fp16` (default)
  2. FP8-FP16: `--fp8 --fp16`
  3. FP8-BF16: `--fp8 --bf16`

- K-V cache can be enabled through `--use-cache`

- Batch size can be changed through `--batch-size=<bs>`

# How to run performance benchmark
The `benchmark` action will run inference with specified input and output sequence lengths multiple times.
```
python3 run.py benchmark GPT3 [frameworks | trt] --variant gpt-5b --working-dir $(pwd)/temp --fp8 --fp16 --batch-size=8 --input-seq-len=128 --output-seq-len=20 --use-cache
```

Expected output for `trt`:
```
***************************
Running 100 iterations with batch size: 8, input sequence length: 128 and output sequence length: 20
  Total Time: 19.05358 s, Average Time: 0.19054 s, 95th Percentile Time: 0.19298 s, 99th Percentile Time: 0.19523 s, Throughput: 839.74 tokens/s
***************************
```

Expected output for `frameworks`:
```
***************************
Running 100 iterations with batch size: 8, input sequence length: 128 and output sequence length: 20
  Total Time: 59.08389 s, Average Time: 0.59084 s, 95th Percentile Time: 0.59606 s, 99th Percentile Time: 0.60624 s, Throughput: 270.80 tokens/s
***************************
```

# How to run accuracy check
The `accuracy` action will run accuracy check on a dataset. Default is to use [LAMBADA](https://paperswithcode.com/dataset/lambada) dataset.
```
python3 run.py accuracy GPT3 [frameworks | trt] --variant gpt-5b --working-dir $(pwd)/temp --fp8 --fp16
```

Expected output for `trt`:
```
***************************
Lambada ppl: 4.7984961574172145, acc(top1): 0.6846497186105182, acc(top3): 0.8649330487094896, acc(top5): 0.9146128468853095
***************************
```

Expected output for `frameworks`:
```
***************************
Lambada ppl: 4.789651459845271, acc(top1): 0.6846497186105182, acc(top3): 0.8649330487094896, acc(top5): 0.9146128468853095
***************************
```
