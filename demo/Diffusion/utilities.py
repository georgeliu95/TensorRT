#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import OrderedDict
from copy import copy
import numpy as np
import os
from PIL import Image
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util
from polygraphy import cuda
import random
from scipy import integrate
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Engine():
    def __init__(
        self,
        model_name,
        engine_dir,
    ):
        self.engine_path =  os.path.join(engine_dir, model_name+'.plan')
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray) ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(self, onnx_path, fp16, input_profile=None):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])
        engine = engine_from_network(network_from_onnx_path(onnx_path), config=CreateConfig(fp16=fp16, profiles=[p]))
        save_engine(engine, path=self.engine_path)

    def activate(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt_util.np_dtype_from_trt(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            # Workaround to convert np dtype to torch
            np_type_tensor = np.empty(shape=[], dtype=dtype)
            torch_type_tensor = torch.from_numpy(np_type_tensor)
            tensor = torch.empty(tuple(shape), dtype=torch_type_tensor.dtype).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")
            exit()

        return self.tensors

class Scheduler():
    def __init__(
        self,
        device = 'cuda',
        beta_start = 0.00085,
        beta_end = 0.012,
        num_train_timesteps = 1000,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.order = 4

        self.beta_start = beta_start
        self.beta_end = beta_end
        betas = (torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=torch.float32) ** 2)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # standard deviation of the initial noise distribution
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.init_noise_sigma = sigmas.max()

        self.device = device

    def set_timesteps(self, steps):
        self.num_inference_steps = steps

        timesteps = np.linspace(0, self.num_train_timesteps - 1, steps, dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=self.device)

        # Move all timesteps to correct device beforehand
        self.timesteps = torch.from_numpy(timesteps).to(device=self.device).float()
        self.derivatives = []

    def get_latent_scales(self):
        input_scales = [1./((sigma**2 + 1) ** 0.5) for sigma in self.sigmas]
        return input_scales

    def get_lms_coefficients(self):
        order = self.order
        lms_coeffs = []

        def get_lms_coefficient(order, t, current_order):
            """
            Compute a linear multistep coefficient.
            """
            def lms_derivative(tau):
                prod = 1.0
                for k in range(order):
                    if current_order == k:
                        continue
                    prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
                return prod
            integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]
            return integrated_coeff

        for step_index in range(self.num_inference_steps):
            order = min(step_index + 1, order)
            lms_coeffs.append([get_lms_coefficient(order, step_index, curr_order) for curr_order in range(order)])
        return lms_coeffs

def save_image(images, image_path_dir, image_name_prefix):
    """
    Save the generated images to png files.
    """
    images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    for i in range(images.shape[0]):
        image_path  = os.path.join(image_path_dir, image_name_prefix+str(i+1)+'-'+str(random.randint(1000,9999))+'.png')
        print(f"Saving image {i+1} / {images.shape[0]} to: {image_path}")
        Image.fromarray(images[i]).save(image_path)
