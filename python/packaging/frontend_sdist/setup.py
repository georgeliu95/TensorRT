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

import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.install import install

tensorrt_module = "##TENSORRT_MODULE##"
tensorrt_version = "##TENSORRT_PYTHON_VERSION##"
tensorrt_submodules = [
    "{}_libs=={}".format(tensorrt_module, tensorrt_version),
    "{}_bindings=={}".format(tensorrt_module, tensorrt_version),
]
nvidia_pip_index_url = os.environ.get("NVIDIA_PIP_INDEX_URL", "https://pypi.nvidia.com")
disable_internal_pip = os.environ.get("NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP", False)


def run_pip_command(args, call_func):
    try:
        return call_func([sys.executable, "-m", "pip"] + args)
    except:
        return call_func([os.path.join(sys.exec_prefix, "bin", "pip")] + args)


class InstallCommand(install):
    def run(self):
        # pip-inside-pip hack ref #3080
        run_pip_command(
            [
                "install",
                "--extra-index-url",
                nvidia_pip_index_url,
                *tensorrt_submodules,
            ],
            subprocess.check_call,
        )

        super().run()


def pip_config_list():
    """Get the current pip config (env vars, config file, etc)."""
    return run_pip_command(["config", "list"], subprocess.check_output).decode()


def parent_command_line():
    """Get the command line of the parent PID."""
    pid = os.getppid()
    # try retrieval using psutil
    try:
        import psutil

        return " ".join(psutil.Process(pid).cmdline())
    except:
        pass
    # fall back to shell
    try:
        return subprocess.check_output(["ps", "-p", str(pid), "-o", "command", "--no-headers"]).decode()
    except:
        return ""


# use pip-inside-pip hack only if the nvidia index is not set in the environment
if disable_internal_pip or nvidia_pip_index_url in pip_config_list() or nvidia_pip_index_url in parent_command_line():
    install_requires = tensorrt_submodules
    cmdclass = {}
else:
    install_requires = []
    cmdclass = {"install": InstallCommand}


setup(
    name=tensorrt_module,
    version=tensorrt_version,
    description="A high performance deep learning inference library",
    long_description="""
NVIDIA TensorRT is an SDK that facilitates high-performance machine learning inference. It is designed to work in a complementary fashion with training frameworks such as TensorFlow, PyTorch, and MXNet. It focuses specifically on running an already-trained network quickly and efficiently on NVIDIA hardware.

**IMPORTANT:** This is a special release of TensorRT designed to work only with TensorRT-LLM. 
Please refrain from upgrading to this version if you are not using TensorRT-LLM. 

To install, please execute the following:
```
pip install tensorrt --extra-index-url {}
```
Or add the index URL to the (space-separated) PIP_EXTRA_INDEX_URL environment variable:
```
export PIP_EXTRA_INDEX_URL='{}'
pip install tensorrt
```
When the extra index url does not contain `{}`, a nested `pip install` will run with the proper extra index url hard-coded.
""".format(
        nvidia_pip_index_url, nvidia_pip_index_url, nvidia_pip_index_url
    ),
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    license="Proprietary",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=[tensorrt_module],
    install_requires=install_requires,
    python_requires=">=3.6",  # ref https://pypi.nvidia.com/tensorrt-bindings/
    cmdclass=cmdclass,
    extras_require={"numpy": "numpy"},
    package_data={tensorrt_module: ["*.so*", "*.pyd", "*.pdb", "*.dll*"]},
    include_package_data=True,
    zip_safe=True,
    keywords="nvidia tensorrt deeplearning inference",
    url="https://developer.nvidia.com/tensorrt",
    download_url="https://github.com/nvidia/tensorrt/tags",
)
