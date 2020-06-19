# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

ARG CUDA_VERSION=10.2
ARG UBUNTU_VERSION=18.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn7-devel-ubuntu${UBUNTU_VERSION}

LABEL maintainer="NVIDIA CORPORATION"

ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown -R trtuser:trtuser /workspace
# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
		build-essential \
		g++-powerpc64le-linux-gnu

RUN cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Set environment and working directory
ENV TRT_RELEASE /tensorrt
ENV TRT_SOURCE /workspace/TensorRT

# Install Cuda 11.0
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
		wget http://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.1-450.36.06-1_amd64.deb && \
		dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.1-450.36.06-1_amd64.deb && \
		apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub && \
		apt-get update && \
		apt-get -y install cuda

# Make Targets Dir
RUN mkdir /usr/local/cuda-11.0/targets/ppc64le-linux && \
    mkdir /usr/local/cuda-11.0/targets/ppc64le-linux/include  && \
    mkdir /usr/local/cuda-11.0/targets/ppc64le-linux/lib/


# Download ppc Cudnn, Cublas, Cudart, and RT
RUN	 wget http://cuda-repo/release-candidates/Libraries/cuDNN/v8.0/8.0.2.5_20200617_28575977/11.0.x-r445/Installer/Ubuntu18_04-ppc64le/libcudnn8_8.0.2.5-1+cuda11.0_ppc64el.deb && \
	 wget http://cuda-repo/release-candidates/Libraries/cuDNN/v8.0/8.0.2.5_20200617_28575977/11.0.x-r445/Installer/Ubuntu18_04-ppc64le/libcudnn8-dev_8.0.2.5-1+cuda11.0_ppc64el.deb && \
	 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/libcublas-dev-11-0_11.0.0.191-1_ppc64el.deb && \
	 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/libcublas-11-0_11.0.0.191-1_ppc64el.deb && \
	 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-cudart-11-0_11.0.171-1_ppc64el.deb && \
	 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-cudart-dev-11-0_11.0.171-1_ppc64el.deb && \
	 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-nvrtc-11-0_11.0.167-1_ppc64el.deb && \
	 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-nvrtc-dev-11-0_11.0.167-1_ppc64el.deb

# Unpack cudnn
RUN dpkg -x libcudnn8_8.0.2.5-1+cuda11.0_ppc64el.deb / && \
    dpkg -x libcudnn8-dev_8.0.2.5-1+cuda11.0_ppc64el.deb /

# Unpack Cublas
RUN dpkg -x libcublas-11-0_11.0.0.191-1_ppc64el.deb cublas && \
    dpkg -x libcublas-dev-11-0_11.0.0.191-1_ppc64el.deb cublas && \
		cp -r cublas/usr/local/cuda-11.0/targets/ppc64le-linux/lib/* /usr/local/cuda-11.0/targets/ppc64le-linux/lib && \
		cp cublas/usr/local/cuda-11.0/include/* /usr/local/cuda-11.0/targets/ppc64le-linux/include/

# Unpack Cudart
RUN dpkg -x cuda-cudart-11-0_11.0.171-1_ppc64el.deb cudart && \
    dpkg -x cuda-cudart-dev-11-0_11.0.171-1_ppc64el.deb cudart && \
		cp cudart/usr/local/cuda-11.0/targets/ppc64le-linux/lib/* /usr/local/cuda-11.0/targets/ppc64le-linux/lib/ && \
		cp -r cudart/usr/local/cuda-11.0/targets/ppc64le-linux/include/* /usr/local/cuda-11.0/targets/ppc64le-linux/include

# Unpack RT
RUN dpkg -x cuda-nvrtc-11-0_11.0.167-1_ppc64el.deb rt && \
    dpkg -x cuda-nvrtc-dev-11-0_11.0.167-1_ppc64el.deb rt && \
		cp rt/usr/local/cuda-11.0/targets/ppc64le-linux/lib/*.so* /usr/local/cuda-11.0/targets/ppc64le-linux/lib/ && \
		cp rt/usr/local/cuda-11.0/targets/ppc64le-linux/lib/stubs/* /usr/local/cuda-11.0/targets/ppc64le-linux/lib/stubs && \
		cp rt/usr/local/cuda-11.0/targets/ppc64le-linux/include/*  /usr/local/cuda-11.0/targets/ppc64le-linux/include

RUN rm -rf cublas cudart rt

WORKDIR /workspace
USER trtuser
RUN ["/bin/bash"]
