#!/bin/bash
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
if [ "$#" -ge 1 ]; then
    echo "Usage: source install.sh [dependencies_dir]";
    exit 1;
elif [ "$#" -eq 1 ]; then
    DEPENDENCIES_DIR=${1};
fi
echo "Using ${DEPENDENCIES_DIR} to store dependencies.";
mkdir -p ${DEPENDENCIES_DIR};

pip install --upgrade pip

# install pytorch
has_torch=$(pip list | grep torch -o | sort -u | awk '{print $1}' | awk '{print length}');
if [ "$has_torch" != "5" ];
then
    echo " > Installing PyTorch...";
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
fi

echo " > Installing Requirements.txt...";
pip install nvidia-pyindex
pip install -r requirements.txt

BASE_DIR=$(pwd);
cd ${DEPENDENCIES_DIR};

# install apex
has_apex=$(pip list | grep apex | awk '{print $1}' | awk '{print length}');
if [ "$has_apex" != "4" ];
then
    echo " > Installing Apex...";
    if [ ! -d "apex" ];
    then
        git clone https://github.com/NVIDIA/apex.git;
    fi
    cd apex
    git config --global --add safe.directory $(pwd)
    git checkout 5b5d41034b506591a316c308c3d2cd14d5187e23
    git apply ${BASE_DIR}/apex.patch # Bypass CUDA version check in apex
    torchcppext=$(pip show torch | grep Location | cut -d' ' -f2)"/torch/utils/cpp_extension.py"
    if [ !-f $torchcppext ];
    then
        echo "Could not locate torch installation using pip";
        exit 1;
    fi
    sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))/pass/' $torchcppext # Bypass CUDA version check in torch
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./
    cd ../
    export PYTHONPATH=$(pwd)/apex/:${PYTHONPATH}
fi

echo " > Installing Megatron-LM...";
if [ ! -d "Megatron-LM" ];
then
    git clone -b main https://github.com/NVIDIA/Megatron-LM.git
fi

cd Megatron-LM
git config --global --add safe.directory $(pwd)
git checkout 992da75a1fd90989eb1a97be8d9ff3eca993aa83
pip install ./
cd ../
export PYTHONPATH=$(pwd)/Megatron-LM/:${PYTHONPATH}

echo " > Installing TransformerEngine...";
MAKEFLAGS="-j6" pip install flash-attn==1.0.2 # explicitly specify version to avoid CUDA version error
MAKEFLAGS="-j6" pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@215dfe7e5bd326cc0a774c3f2149a0acc41535c4

echo " > Installing NeMo...";
if [ ! -d "NeMo" ];
then
    git clone -b main https://github.com/NVIDIA/NeMo.git
fi
cd NeMo
git config --global --add safe.directory $(pwd)
git checkout bf270794267e0240d8a8b2f2514c80c6929c76f1
bash reinstall.sh
cd ../
export PYTHONPATH=$(pwd)/NeMo/:${PYTHONPATH}
cd ../
