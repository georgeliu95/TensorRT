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

ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=7

# TODO: Update - unused in 22.09
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-centos${OS_VERSION} 
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 8.5.0.9
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG wheel trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install requried packages
RUN yum -y groupinstall "Development Tools"
RUN yum -y install \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    wget \
    perl-core \
    git \
    pkg-config \
    unzip \
    sudo

# Install python3
RUN yum install -y python36 python3-devel

# Install TensorRT
COPY docker_qa/downloadInternal.py /tmp/downloadInternal.py
RUN python3 /tmp/downloadInternal.py --cuda $CUDA_VERSION --os 7

# Install dev-toolset-8 for g++ version that supports c++14
RUN yum -y install centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum -y install devtoolset-8

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
RUN pip3 install numpy
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter jupyterlab

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc && rm ngccli_cat_linux.zip ngc.md5 && echo "no-apikey\nascii\n" | ngc config set

RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
# Use devtoolset-8 as default compiler
ENV PATH="/opt/rh/devtoolset-8/root/bin:${PATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]