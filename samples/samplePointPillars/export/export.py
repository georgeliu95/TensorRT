#
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
#

import argparse
import glob
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from numpy import *
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import onnx
from onnxsim import simplify
import os, sys
from simplifier_onnx import simplify_onnx as simplify_onnx


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class ExportablePFNLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        inputs_shape = inputs.cpu().detach().numpy().shape
        if len(inputs_shape) == 4:
            inputs = inputs.view((-1, inputs_shape[2], inputs_shape[3]))
        x = self.model.linear(inputs)
        voxel_num_points = inputs_shape[-2]
        if self.model.use_norm:
            x = self.model.norm(x.permute(0, 2, 1))
            x = F.relu(x)
            x = F.max_pool1d(x, voxel_num_points, stride=1)
            x_max = x.permute(0, 2, 1)
        else:
            x = F.relu(x)
            x = x.permute(0, 2, 1)
            x = F.max_pool1d(x, voxel_num_points, stride=1)
            x_max = x.permute(0, 2, 1)
        if len(inputs_shape) == 4:
            x_max_shape = x_max.cpu().detach().numpy().shape
            x_max = x_max.view((-1, inputs_shape[1], x_max_shape[2]))
        else:
            x_max = x_max.squeeze(1)
        if self.model.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class ExportablePillarVFE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxel_features, voxel_num_points, coords):
        points_mean = voxel_features[..., :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[..., :3] - points_mean
        f_center = torch.zeros_like(voxel_features[..., :3])
        f_center[..., 0] = voxel_features[..., 0] - (coords[..., 3].to(voxel_features.dtype).unsqueeze(2) * self.model.voxel_x + self.model.x_offset)
        f_center[..., 1] = voxel_features[..., 1] - (coords[..., 2].to(voxel_features.dtype).unsqueeze(2) * self.model.voxel_y + self.model.y_offset)
        f_center[..., 2] = voxel_features[..., 2] - (coords[..., 1].to(voxel_features.dtype).unsqueeze(2) * self.model.voxel_z + self.model.z_offset)
        if self.model.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.model.with_distance:
            points_dist = torch.norm(voxel_features[..., :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        for pfn in self.model.pfn_layers:
            exportable_pfn = ExportablePFNLayer(pfn)
            features = exportable_pfn(features)
        return features


class ExportableScatter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pillar_features, coords):
        batch_spatial_features = []
        batch_size = coords[..., 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.model.num_bev_features,
                self.model.nz * self.model.nx * self.model.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            batch_mask = coords[batch_idx, :, 0] == batch_idx
            this_coords = coords[batch_idx, batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.model.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_idx, batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(
            -1, self.model.num_bev_features * self.model.nz,
            self.model.ny, self.model.nx
        )
        return batch_spatial_features


class ExportableBEVBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spatial_features):
        ups = []
        x = spatial_features
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)
            if len(self.model.deblocks) > 0:
                ups.append(self.model.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.model.deblocks) > len(self.model.blocks):
            x = self.model.deblocks[-1](x)
        return x


class ExportableAnchorHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spatial_features_2d, batch_size):
        cls_preds = self.model.conv_cls(spatial_features_2d)
        box_preds = self.model.conv_box(spatial_features_2d)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        if self.model.conv_dir_cls is not None:
            dir_cls_preds = self.model.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None
        return cls_preds, box_preds, dir_cls_preds


class ExportablePointPillar(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module_list = model.module_list
        self.exportable_vfe = ExportablePillarVFE(self.module_list[0])
        self.exportable_scatter = ExportableScatter(self.module_list[1])
        self.exportable_bev_backbone = ExportableBEVBackbone(self.module_list[2])
        self.exportable_anchor_head = ExportableAnchorHead(self.module_list[3])

    def forward(self, voxel_features, voxel_num_points, coords):
        self.batch_size = 1
        pillar_features = self.exportable_vfe(voxel_features, voxel_num_points, coords) #"PillarVFE"
        spatial_features = self.exportable_scatter(pillar_features, coords) #"PointPillarScatter"
        spatial_features_2d = self.exportable_bev_backbone(spatial_features) #"BaseBEVBackbone"
        cls_preds, box_preds, dir_cls_preds = self.exportable_anchor_head(spatial_features_2d, self.batch_size) #"AnchorHeadSingle"
        return cls_preds, box_preds, dir_cls_preds


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--cfg_file',
        type=str,
        default='cfgs/kitti_models/pointpillar.yaml',
        help='specify the config for demo'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='demo_data',
        help='specify the point cloud data file or directory'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='specify the pretrained model'
    )
    parser.add_argument(
        "--output_model",
        '-o',
        type=str,
        required=False,
        default=None,
        help="Path to save the exported ONNX model"
    )
    parser.add_argument(
        '--ext',
        type=str,
        default='.bin',
        help='specify the extension of your point cloud data file'
    )
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('------ Convert OpenPCDet model to ONNX ------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model = ExportablePointPillar(model)
    model.cuda()
    model.eval()
    with torch.no_grad():
        MAX_VOXELS = 10000
        MAX_POINTS = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL
        dummy_voxel_features = torch.zeros(
            (1, MAX_VOXELS, MAX_POINTS, 4),
            dtype=torch.float32,
            device='cuda:0'
        )
        dummy_voxel_num_points = torch.zeros(
            (1, MAX_VOXELS,),
            dtype=torch.int32,
            device='cuda:0'
        )
        dummy_coords = torch.zeros(
            # 4: (batch_idx, x, y, z)
            (1, MAX_VOXELS, 4),
            dtype=torch.int32,
            device='cuda:0'
        )
        _, temp_onnx = tempfile.mkstemp(".onnx")
        torch.onnx.export(
            model,
            (dummy_voxel_features, dummy_voxel_num_points, dummy_coords),
            temp_onnx,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            input_names = ['input', 'voxel_num_points', 'coords'],
            output_names = ['cls_preds', 'box_preds', 'dir_cls_preds'],
            dynamic_axes={
                "input": {0: "batch"},
                "voxel_num_points": {0: "batch"},
                "coords": {0: "batch"}
            }
        )
        onnx_model = onnx.load(temp_onnx)
        model_simp, check = simplify(
            onnx_model,
            dynamic_input_shape=True,
            input_shapes={
                "input": (1, MAX_VOXELS, MAX_POINTS, 4),
                'voxel_num_points': (1, MAX_VOXELS),
                'coords': (1, MAX_VOXELS, 4)
            }
        )
        assert check, "Failed on simplifying the ONNX model"
        model_simp = simplify_onnx(model_simp, cfg)
        onnx.save(model_simp, args.output_model)
        os.remove(temp_onnx)
    logger.info(f'Model exported to {args.output_model}')


if __name__ == '__main__':
    main()
