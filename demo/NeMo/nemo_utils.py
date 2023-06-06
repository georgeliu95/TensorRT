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

import gc
import os
import torch
from omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

def load_nemo_model(cfg, model_class=MegatronGPTModel):
    if not cfg.gpt_model_file:
        raise ValueError("Need to provide a nemo gpt model through config gpt_model_file.")

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.gpt_model_file):
        save_restore_connector.model_extracted_dir = cfg.gpt_model_file

    pretrained_cfg = MegatronGPTModel.restore_from(
        restore_path=cfg.gpt_model_file,
        trainer=trainer,
        return_config=True,
        save_restore_connector=save_restore_connector,
    )
    OmegaConf.set_struct(pretrained_cfg, True)
    with open_dict(pretrained_cfg):
        pretrained_cfg.sequence_parallel = False
        pretrained_cfg.activations_checkpoint_granularity = None
        pretrained_cfg.activations_checkpoint_method = None
        pretrained_cfg.precision = trainer.precision
        if trainer.precision == "16":
            pretrained_cfg.megatron_amp_O2 = False
    model = model_class.restore_from(
        restore_path=cfg.gpt_model_file,
        trainer=trainer,
        override_config_path=pretrained_cfg,
        save_restore_connector=save_restore_connector,
    )

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    model.eval()
    return model.cuda()

def release_nemo_model(model):
    print(f"Releaseing nemo model.")
    model.model.cpu()
    del model.model
    gc.collect()
    torch.cuda.empty_cache()
    model.model = None
