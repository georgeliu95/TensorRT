# TensorRT FP8 Inference for NeMo models
This repository demonstrates TensorRT inference with NeMo Megatron models in FP8 precision.

Currently, this repository supports NeMo GPT models only.

# Environment Setup
It's recommended to run inside a container to avoid conflicts in dependencies. Tested with TensorRT container [`tensorrt:23.04-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags) and PyTorch container [`pytorch:23.03-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags). A GPU with compute capability 9.0 or above is required to run the demo.

```
source install.sh
```

> The script will install required dependencies and it can take more than 30 minutes.


## Download NeMo model
A GPT-5B model trained with FP16 precision can be downloaded with:
```
wget https://huggingface.co/nvidia/nemo-megatron-gpt-5B/resolve/main/nemo_gpt5B_fp16_tp1.nemo
```

Please get a NeMo GPT model trained with FP8 precision to be compared with FP16 precision.

# Export NeMo to ONNX and TensorRT

Assume `5b_fp8_tp1.nemo` is a NeMo GPT-5B trained with FP8 precision. we can run below command to export the NeMo model to a TRT engine. The ONNX model is used as a middle step, and it will be stored at the specified path.
```
# python export.py gpt_model_file=<INPUT_NEMO_MODEL> onnx_model_file=<OUTPUT_ONNX_NAME> trt_engine_file=<OUTPUT_TRT_NAME>
mkdir onnx
python export.py gpt_model_file=5b_fp8_tp1.nemo onnx_model_file=onnx/5b_fp8_tp1.onnx trt_engine_file=5b_fp8_tp1.plan
```

# Run E2E inference
Perform end-to-end inference on an input prompt with TRT:
```
python main.py runtime=trt trt_engine_file=5b_fp8_tp1.plan prompts=["Tell me an interesting fact about TensorRT."]
```

Expected output:
```
***************************
{'sentences': ['Tell me an interesting fact about TensorRT.\n\nTensorRT is a library for training and deploying neural networks. It is written in C++ and is available on Windows, Linux, and'],
 'tokens': [['<|endoftext|>', 'Tell', ' me', ' an', ' interesting', ' fact', ' about', ' T', 'ensor', 'RT', '.', '\n', '\n', 'T', 'ensor', 'RT', ' is', ' a', ' library', ' for', ' training', ' and', ' deploying', ' neural', ' networks', '.', ' It', ' is', ' written', ' in', ' C', '++', ' and', ' is', ' available', ' on', ' Windows', ',', ' Linux', ',', ' and']],
 'logprob': tensor([[-8.8471, -1.9896, -7.4929, -1.8296, -0.7938, -0.4374, -7.7575, -6.3849,
         -7.1476, -1.3660, -0.4079, -0.1451, -2.2410, -0.0183, -0.0904, -0.4929,
         -0.6474, -2.7981, -1.0308, -2.4455, -1.0503, -0.8344, -1.3654, -0.1558,
         -1.0352, -1.1387, -1.5104, -1.7296, -0.2362, -0.1349, -0.0814, -0.5735,
         -1.3795, -2.2406, -1.2948, -1.4569, -0.1648, -0.7849, -0.3062, -0.3810]], device='cuda:0'),
 'full_logprob': None, 'token_ids': [[50256, 24446, 502, 281, 3499, 1109, 546, 309, 22854, 14181, 13, 198, 198, 51, 22854, 14181, 318, 257, 5888, 329, 3047, 290, 29682, 17019, 7686, 13, 632, 318, 3194, 287, 327, 4880, 290, 318, 1695, 319, 3964, 11, 7020, 11, 290]], 'offsets': [[0, 0, 4, 7, 10, 22, 27, 33, 35, 40, 42, 43, 44, 45, 46, 51, 53, 56, 58, 66, 70, 79, 83, 93, 100, 109, 110, 113, 116, 124, 127, 129, 131, 135, 138, 148, 151, 159, 160, 166, 167]]}
***************************
```

Perform end-to-end inference on an input prompt with NeMo:
```
python main.py runtime=nemo gpt_model_file=5b_fp8_tp1.nemo prompts=["Tell me an interesting fact about TensorRT."]
```

Expected output:
```
***************************
{'sentences': ['Tell me an interesting fact about TensorRT.\n\nTensorRT is a library for training and deploying neural networks. It is written in C++ and is available on Windows, Linux, and'],
 'tokens': [['<|endoftext|>', 'Tell', ' me', ' an', ' interesting', ' fact', ' about', ' T', 'ensor', 'RT', '.', '\n', '\n', 'T', 'ensor', 'RT', ' is', ' a', ' library', ' for', ' training', ' and', ' deploying', ' neural', ' networks', '.', ' It', ' is', ' written', ' in', ' C', '++', ' and', ' is', ' available', ' on', ' Windows', ',', ' Linux', ',', ' and']],
 'logprob': tensor([[-8.9012, -1.9124, -7.5838, -1.9213, -0.8400, -0.4180, -7.7495, -6.4685,
         -7.3179, -1.4008, -0.3859, -0.1656, -2.1336, -0.0182, -0.0730, -0.4786,
         -0.6357, -2.7409, -1.0404, -2.4107, -1.0784, -0.9290, -1.4114, -0.1634,
         -1.0954, -1.1354, -1.5088, -1.5949, -0.2563, -0.1336, -0.0771, -0.5719,
         -1.4210, -2.2415, -1.2126, -1.5932, -0.1459, -0.7712, -0.3292, -0.4084]], device='cuda:0'),
 'full_logprob': None, 'token_ids': [[50256, 24446, 502, 281, 3499, 1109, 546, 309, 22854, 14181, 13, 198, 198, 51, 22854, 14181, 318, 257, 5888, 329, 3047, 290, 29682, 17019, 7686, 13, 632, 318, 3194, 287, 327, 4880, 290, 318, 1695, 319, 3964, 11, 7020, 11, 290]], 'offsets': [[0, 0, 4, 7, 10, 22, 27, 33, 35, 40, 42, 43, 44, 45, 46, 51, 53, 56, 58, 66, 70, 79, 83, 93, 100, 109, 110, 113, 116, 124, 127, 129, 131, 135, 138, 148, 151, 159, 160, 166, 167]]}
***************************
```

# Accuracy

## Evaluating Sequence Perplexity Using The LAMBADA Dataset
Perform accuracy check on GPT-5B FP8 model with TRT:
```
python main.py runtime=trt trt_engine_file=5b_fp8_tp1.plan mode=accuracy
```

Expected output:
```
***************************
Lambada ppl: 4.7984961574172145
***************************
```

Perform accuracy check on GPT-5B FP8 model with NeMo:
```
python main.py runtime=nemo gpt_model_file=5b_fp8_tp1.nemo mode=accuracy
```

Expected output:
```
***************************
Lambada ppl: 4.789651459845271
***************************
```

# Benchmark

## Evaluating performance with fixed input and output sequence length
Perform benchmark with input sequence 128 and output sequence 20 with TRT, use `batch_size=8`:
```
python3 export.py gpt_model_file=5b_fp8_tp1.nemo trt_engine_file=5b_fp8_tp1.plan batch_size=8
python3 main.py runtime=trt trt_engine_file=5b_fp8_tp1.plan batch_size=8 mode=benchmark benchmark.loop=100 benchmark.warm_up=1 benchmark.input_seq_len=128 benchmark.output_seq_len=20
```

Expected output:
```
***************************
Running 100 iterations with batch size: 8, input sequence length: 128 and output sequence length: 20
  Total Time: 19.05358 s, Average Time: 0.19054 s, 95th Percentile Time: 0.19298 s, 99th Percentile Time: 0.19523 s, Throughput: 839.74 tokens/s
***************************
```

Perform benchmark with input sequence 128 and output sequence 20 with NeMo (NeMo FP8 benchmark mode requires batch size to be a multiple of 8.):
```
python3 main.py runtime=nemo gpt_model_file=5b_fp8_tp1.nemo batch_size=8 mode=benchmark benchmark.loop=100 benchmark.warm_up=1 benchmark.input_seq_len=128 benchmark.output_seq_len=20
```

Expected output:
```
***************************
Running 100 iterations with batch size: 8, input sequence length: 128 and output sequence length: 20
  Total Time: 59.08389 s, Average Time: 0.59084 s, 95th Percentile Time: 0.59606 s, 99th Percentile Time: 0.60624 s, Throughput: 270.80 tokens/s
***************************
```
