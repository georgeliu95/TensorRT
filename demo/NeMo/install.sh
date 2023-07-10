#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

DEPENDENCIES_DIR="temp";
if [ "$#" -gt 1 ]; then
    echo "Usage: source install.sh [dependencies_dir]";
    return 0;
elif [ "$#" -eq 1 ]; then
    DEPENDENCIES_DIR=${1};
fi
echo "Using '$(pwd)/${DEPENDENCIES_DIR}' to store dependencies.";
mkdir -p "${DEPENDENCIES_DIR}";

pip install --upgrade pip

echo " > Installing Requirements.txt...";
pip install nvidia-pyindex || { echo "Could not install nvidia-pyindex, stopping install"; exit 1; }
# # One of the hidden dependencies require Cython, but doesn't specify it.
# # https://github.com/VKCOM/YouTokenToMe/pull/108
# # WAR by installing Cython before requirements.
pip install "Cython>=0.29.34" || { echo "Could not install Cython, stopping install"; exit 1; }
pip install -r requirements.txt || { echo "Could not install dependencies, stopping install"; exit 1; }

BASE_DIR=$(pwd);
cd "${DEPENDENCIES_DIR}" || exit

# install apex
has_apex=$(pip list | grep "^apex " | grep "apex" -o | awk '{print $1}' | awk '{print length}');
if [ "$has_apex" != "4" ];
then
    echo " > Installing Apex...";
    if [ ! -d "apex" ];
    then
        git clone https://github.com/NVIDIA/apex.git;
    fi
    cd apex || exit
    APEX_PATH="$(pwd)"
    git config --global --add safe.directory "${APEX_PATH}"
    git checkout 5b5d41034b506591a316c308c3d2cd14d5187e23
    git apply "${BASE_DIR}"/apex.patch # Bypass CUDA version check in apex
    torchcppext=$(pip show torch | grep Location | cut -d' ' -f2)"/torch/utils/cpp_extension.py"
    if [ ! -f "$torchcppext" ];
    then
        echo "Could not locate torch installation using pip";
        exit 1;
    fi
    sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))/pass/' "$torchcppext" # Bypass CUDA version check in torch
	unset torchcppext
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./
    cd ../
    export PYTHONPATH="${APEX_PATH}:${PYTHONPATH}"
    unset APEX_PATH
fi
unset has_apex

echo " > Installing Megatron-LM...";
if [ ! -d "Megatron-LM" ];
then
    git clone -b main https://github.com/NVIDIA/Megatron-LM.git
fi

cd Megatron-LM || exit
MEGATRON_PATH="$(pwd)"
git config --global --add safe.directory "${MEGATRON_PATH}"
git checkout 992da75a1fd90989eb1a97be8d9ff3eca993aa83
pip install ./
cd ../
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"
unset MEGATRON_PATH

echo " > Installing TransformerEngine...";
MAKEFLAGS="-j6" pip install flash-attn==1.0.6 --no-build-isolation # explicitly specify version to avoid CUDA version error
MAKEFLAGS="-j6" pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@804f120322a13cd5f21ea8268860607dcecd055c

echo " > Patching TransformerEngine...";
te_loc="$(pip show transformer_engine | grep '^Location' | awk '{print $2}')"
cd "${te_loc}/transformer_engine" || { echo "Could not locate transformer engine install path"; exit 1; }
# Use sys.executable when calling pip within subprocess to recognize virtualenv.
# If patch is already applied, skip it and proceed with the rest of the script, quit otherwise.
# NOTE: patch needs to be updated to track the current version of TE installed above.
OUT="$(patch --forward common/__init__.py < "${BASE_DIR}"/transformer_engine.patch)" || echo "${OUT}" | grep "Skipping patch" -q || { echo "Could not patch transformer engine because ${OUT}"; exit 1; }
cd - || exit
unset te_loc

echo " > Installing NeMo...";
if [ ! -d "NeMo" ];
then
    git clone -b main https://github.com/NVIDIA/NeMo.git
fi
cd NeMo || exit
NeMo_PATH="$(pwd)"
git config --global --add safe.directory "${NeMo_PATH}"
git checkout bf270794267e0240d8a8b2f2514c80c6929c76f1
bash reinstall.sh
cd ../
export PYTHONPATH="${NeMo_PATH}:${PYTHONPATH}"
unset NeMo_PATH

if [ ! -f "GPT3/convert_te_onnx_to_trt_onnx.py" ];
then
    echo " > Copying opset19 conversion script...";
    if [ ! -f "../../../scripts/convert_te_onnx_to_trt_onnx.py" ];
    then
        echo "Opset19 conversion script is not located at <ROOT_DIR>/scripts/convert_te_onnx_to_trt_onnx.py";
        return 1;
    fi
    cp ../../../scripts/convert_te_onnx_to_trt_onnx.py ../GPT3/convert_te_onnx_to_trt_onnx.py
fi

cd ../

unset BASE_DIR
unset DEPENDENCIES_DIR