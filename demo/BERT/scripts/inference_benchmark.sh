#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage: run_benchmark(batch_sizes, model_variant: (base/large), precision: (fp16/fp32), sequence_length, max_batch_size)
run_benchmark() {
BATCH_SIZES="${1}"

MODEL_VARIANT="${2}"
PRECISION="${3}"
SEQUENCE_LENGTH="${4}"
MAX_BATCH="${5}"

CHECKPOINTS_DIR="/workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_${MODEL_VARIANT}_${PRECISION}_${SEQUENCE_LENGTH}_v2"
ENGINE_NAME="/workspace/TensorRT/demo/BERT/engines/bert_${MODEL_VARIANT}_${PRECISION}_bs${MAX_BATCH}_seqlen${SEQUENCE_LENGTH}_benchmark.engine"

echo "==== Benchmarking BERT ${MODEL_VARIANT} ${PRECISION} SEQLEN ${SEQUENCE_LENGTH} ===="
if [ ! -f ${ENGINE_NAME} ]; then
    if [ ! -d ${CHECKPOINTS_DIR} ]; then
        echo "Downloading checkpoints: scripts/download_model.sh ${MODEL_VARIANT} ${PRECISION} ${SEQUENCE_LENGTH}"
        scripts/download_model.sh "${MODEL_VARIANT}" "${PRECISION}" "${SEQUENCE_LENGTH}"
    fi;

    echo "Building engine: python3 builder.py -m ${CHECKPOINTS_DIR}/model.ckpt-8144 -o ${ENGINE_NAME} ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} --${PRECISION} -c ${CHECKPOINTS_DIR}"
    python3 builder.py -m ${CHECKPOINTS_DIR}/model.ckpt-8144 -o ${ENGINE_NAME} ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} --${PRECISION} -c ${CHECKPOINTS_DIR}
fi;

python3 perf.py ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -e ${ENGINE_NAME}
echo
}

mkdir -p /workspace/TensorRT/demo/BERT/engines

# BERT BASE
## FP16
run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp16" "128" "32"
run_benchmark "-b 64" "base" "fp16" "128" "64"
run_benchmark "-b 128" "base" "fp16" "128" "128"

run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp16" "384" "32"
run_benchmark "-b 64" "base" "fp16" "384" "64"
run_benchmark "-b 128" "base" "fp16" "384" "128"

## FP32
run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp32" "128" "32"
run_benchmark "-b 64" "base" "fp32" "128" "64"
run_benchmark "-b 128" "base" "fp32" "128" "128"

run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp32" "384" "32"
run_benchmark "-b 64" "base" "fp32" "384" "64"
run_benchmark "-b 128" "base" "fp32" "384" "128"

# BERT LARGE
## FP16
run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp16" "128" "32"
run_benchmark "-b 64" "large" "fp16" "128" "64"
run_benchmark "-b 128" "large" "fp16" "128" "128"

run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp16" "384" "32"
run_benchmark "-b 64" "large" "fp16" "384" "64"
run_benchmark "-b 128" "large" "fp16" "384" "128"

## FP32
run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp32" "128" "32"
run_benchmark "-b 64" "large" "fp32" "128" "64"
run_benchmark "-b 128" "large" "fp32" "128" "128"

run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp32" "384" "32"
run_benchmark "-b 64" "large" "fp32" "384" "64"
run_benchmark "-b 128" "large" "fp32" "384" "128"
