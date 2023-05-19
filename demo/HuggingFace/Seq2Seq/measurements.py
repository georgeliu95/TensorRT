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

"""
Utils for accuracy check and performance measurements for Seq2Seq models.
"""

# torch
import torch

from NNDF.torch_utils import use_cuda
from NNDF.tensorrt_utils import TRTNativeRunner

@use_cuda
def calculate_perplexity_helper_encoder_decoder(
    encoder,
    decoder,
    tokenizer,
    input_str,
    reference_str,
    batch_size,
    max_length=None,
    use_cuda=True,
):
    input_ids = tokenizer([input_str] * batch_size, padding=True, return_tensors="pt").input_ids
    decoder_input_ids = tokenizer([reference_str] * batch_size, padding=True, return_tensors="pt").input_ids

    if use_cuda:
        input_ids = input_ids.int().cuda()
        decoder_input_ids = decoder_input_ids.cuda()

    encoder_outputs = encoder(input_ids=input_ids)

    # Set the first token to be pad token
    decoder_input_ids_padded = torch.full(
        decoder_input_ids.size()[:-1] + (decoder_input_ids.size()[-1] + 1,),
        tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
        dtype=decoder_input_ids.dtype,
        device=decoder_input_ids.device,
    )
    decoder_input_ids_padded[..., 1:] = decoder_input_ids

    if isinstance(decoder, TRTNativeRunner):
        decoder.set_encoder_hidden_states(encoder_outputs.last_hidden_state)

    with torch.no_grad():
        if max_length is not None:
            decoder_input_ids_padded = decoder_input_ids_padded[:, :max_length]
        
        logits = decoder(
            input_ids=decoder_input_ids_padded,
            encoder_outputs=encoder_outputs
        ).logits

        # Truncate the last prediction
        logits = logits[:, :-1, :]
        loss = torch.nn.CrossEntropyLoss()(logits.permute((0, 2, 1)), decoder_input_ids)
        return torch.exp(loss).item()

@use_cuda
def calculate_perplexity_helper_decoder(
    decoder,
    tokenizer,
    input_str,
    batch_size,
    max_length=None,
    use_cuda=True
):
    
    input_str = input_str.replace("\\n", "\n")
    input_ids = tokenizer([input_str] * batch_size, padding=False, return_tensors="pt").input_ids

    if use_cuda:
        input_ids = input_ids.to("cuda")

    with torch.no_grad():
        if max_length is not None:
            input_ids = input_ids[:, :max_length]
        logits = decoder(input_ids).logits
        # Shift logits and target ids so that probabilities generated by token < n line up with output token n.
        shifted_logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        loss = torch.nn.CrossEntropyLoss()(shifted_logits.permute((0, 2, 1)), target_ids)
        return torch.exp(loss).item()