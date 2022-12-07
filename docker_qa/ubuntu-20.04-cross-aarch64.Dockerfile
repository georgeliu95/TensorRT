# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# This docker file can compile natively for x86_64 Ubuntu and cross compile for
# ARM SBSA Ubuntu
ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=20.04

FROM gitlab-master.nvidia.com:5005/dl/dgx/cuda:11.8-devel-ubuntu20.04--5691963
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 8.5.1.7
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential

# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install cross-compilation toolchains
RUN apt-get install -y g++-8-aarch64-linux-gnu

# Install cross-compilation CUDA packages
RUN wget http://cuda-repo/release-candidates/kitpicks/cuda-r11-8/11.8.0/056/local_installers/cuda-repo-cross-sbsa-ubuntu2004-11-8-local_11.8.0-1_all.deb &&\
    dpkg -i cuda-repo-cross-sbsa-ubuntu2004-11-8-local_11.8.0-1_all.deb &&\
    cp /var/cuda-repo-cross-sbsa-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ &&\
    apt-get update && \
    apt-get -y install cuda-cross-sbsa

# Install cross-cudnn
RUN wget http://cuda-repo/release-candidates/kitpicks/cudnn-v8-6-cuda-11-8/8.6.0.163/001/repos/ubuntu2004/cross-linux-sbsa/libcudnn8-cross-sbsa_8.6.0.163-1+cuda11.8_all.deb &&\
    dpkg -i libcudnn8-cross-sbsa_8.6.0.163-1+cuda11.8_all.deb

# Install cross TensorRT
COPY docker_qa/downloadInternal.py /tmp/downloadInternal.py
RUN python3 /tmp/downloadInternal.py --cuda $CUDA_VERSION --os cross-sbsa

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter jupyterlab
# Workaround to remove numpy installed with tensorflow
RUN pip3 install --upgrade numpy

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
