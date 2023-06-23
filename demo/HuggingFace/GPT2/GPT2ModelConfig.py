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

# Base Class
from Seq2Seq.Seq2SeqModelConfig import Seq2SeqModelTRTConfig

class GPT2ModelTRTConfig(Seq2SeqModelTRTConfig):

    TARGET_MODELS = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/gpt-neox-20b",
        "EleutherAI/gpt-j-6b",
        "cerebras/Cerebras-GPT-111M",
        "cerebras/Cerebras-GPT-256M",
        "cerebras/Cerebras-GPT-1.3B",
        "cerebras/Cerebras-GPT-2.7B",
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        #"facebook/opt-30b", # Too big for single GPU
        #"facebook/opt-66b", # Too big for single GPU
    ]

    def __init__(self, **kwargs):

        super().__init__(
            network_name="GPT2",
            **kwargs
        )

    def from_hf_config(self, hf_config):
        super().from_hf_config(hf_config)
        # Additional parameter to disable HF warning
        self.pad_token_id = self.eos_token_id

        """
        GPT model's n_positions is too long (~2048). The model size for the models in the demo
        is not large enough to generate useful information and therefore will generate repetitive sentences.
        Truncate to 100 for useful informations.
        """
        self.max_length = min(self.max_length, 100)
        self.max_input_length = self.max_length
        self.max_output_length = self.max_length

    def set_generation_config(self, generation_config):
        super().set_generation_config(generation_config)
        self.generation_config.pad_token_id = self.eos_token_id

    def get_metadata_string(self, metadata) -> str:
        # Remove redundant GPT2 name
        metadata = metadata._replace(variant=metadata.variant.replace("GPT2-","").replace("EleutherAI/","").replace("cerebras/","").replace("facebook/",""))
        return super().get_metadata_string(metadata)
