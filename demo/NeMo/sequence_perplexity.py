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

import math
import numpy as np
import torch

__all__ = ['SequencePerplexity']

class SequencePerplexity():
    def __init__(self):
        super().__init__()
        self.ppls = []

    def update(self, ds_input, response, tokenizer):
        batch_size = len(response['token_ids'])
        for batch, tokens in enumerate(response['token_ids']):
            inp_len = ds_input.lens[batch]
            conti_len = ds_input.conti_len[batch]

            response_token_ids = tokens[:inp_len]
            assert response_token_ids == ds_input.inp_enc[batch][:-1], f"Mismatch in input tokens."

            log_probs = response['full_logprob'][batch][:inp_len]
            log_probs = log_probs[-conti_len:]

            conti_token_ids = ds_input.inp_enc[batch][-conti_len:]
            conti_tokens = tokenizer.ids_to_tokens(conti_token_ids)

            log_probs = log_probs.cpu().to(torch.float32)
            conti_enc = torch.tensor(tokenizer.tokens_to_ids(conti_tokens))
            conti_probs = torch.gather(log_probs, 1, conti_enc.unsqueeze(-1)).squeeze(-1)

            ppl = float(conti_probs.sum())
            self.ppls.append(ppl)

    def compute(self):
        return math.exp(-np.mean(np.array(self.ppls)))
