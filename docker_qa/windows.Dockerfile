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

FROM urm.nvidia.com/sw-tensorrt-docker/windows-servercore:latest

# Set PowerShell as the default shell
SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop';"]

# Install Chocolatey
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; \
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; \
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Download Visual Studio, CUDA and cuDNN
RUN (New-Object System.Net.WebClient).DownloadFile('https://aka.ms/vs/17/release/vs_buildtools.exe', 'C:\vs2022_BuildTools.exe')
RUN (New-Object System.Net.WebClient).DownloadFile('https://aka.ms/vs/17/release/channel', 'C:\VisualStudio.17.Release.chman')
RUN (New-Object System.Net.WebClient).DownloadFile('http://cuda-repo.nvidia.com/release-candidates/kitpicks/cuda-r12-0/12.0.0/031/local_installers/cuda_12.0.0_527.41_windows.exe', 'C:\cuda_install.exe')
RUN (New-Object System.Net.WebClient).DownloadFile('http://cuda-repo.nvidia.com/release-candidates/kitpicks/cudnn-v8-8-cuda-12-0/8.8.1.4/001/redist/cudnn/cudnn/windows-x86_64/cudnn-windows-x86_64-8.8.1.4_cuda12-archive.zip', 'C:\cudnn.zip')

# Install Visual Studio VCTools
RUN C:\vs2022_BuildTools.exe --quiet --wait --norestart --nocache install \
        --installPath 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools' \
        --channelUri C:\VisualStudio.17.Release.chman \
        --installChannelUri C:\VisualStudio.17.Release.chman \
        --add Microsoft.VisualStudio.Workload.VCTools \
        --includeRecommended 

RUN Remove-Item -Force 'C:\vs2022_BuildTools.exe'; \
    Remove-Item -Force 'C:\VisualStudio.17.Release.chman'

# Install CUDA (selected components)
RUN Start-Process -Wait -FilePath "C:\cuda_install.exe" -ArgumentList '-s','cuda_profiler_api_12.0 cudart_12.0 nvcc_12.0 nvrtc_12.0 nvrtc_dev_12.0 nvtx_12.0 cublas_12.0 cublas_dev_12.0 thrust_12.0 visual_studio_integration_12.0'; \
    Remove-Item -Force "C:\cuda_install.exe"

# Install CuDNN
RUN Expand-Archive -Path "C:\cudnn.zip" -DestinationPath "C:\cudnn" -Force; \
    $extractedFolderPath = Get-ChildItem -Path "C:\cudnn" | Select-Object -First 1 -ExpandProperty FullName; \
    Move-Item -Path "$extractedFolderPath\*" -Destination "C:\cudnn"; \
    Remove-Item -Path $extractedFolderPath; \
    Remove-Item -Force "C:\cudnn.zip"

# Install CMake and Python using Chocolatey
RUN choco install cmake -y; \
    choco install python --version 3.10.0 -y

# Set environment variables
ENV CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
ENV CUDNN_PATH='C:\cudnn'
ENV VS_ROOT_PATH='C:\Program Files\Microsoft Visual Studio\2022\BuildTools'
ENV MS_BUILD_PATH='C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin'
ENV CMAKE_PATH='C:\Program Files\CMake\bin'

# Copy Visual Studio Integration files
RUN Copy-Item -Path "$env:CUDA_PATH\extras\visual_studio_integration\MSBuildExtensions\*" -Destination 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations' -PassThru

# Set PATH
RUN $env:PATH = $env:CUDA_PATH + ';' + $env:CUDNN_PATH + ';' + $env:MS_BUILD_PATH + ';' + $env:CMAKE_PATH + ';' + $env:PATH; \
    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, [EnvironmentVariableTarget]::Machine)

# Set the default shell to PowerShell
CMD ["powershell"]
