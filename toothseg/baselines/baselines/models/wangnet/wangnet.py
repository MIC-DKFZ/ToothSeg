from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy import ndimage
import torch
from torchtyping import TensorType

from baselines.models.base import EncoderDecoderModule
from baselines.models.cluster import learned_region_cluster
import baselines.nn as nn


class WangNet(EncoderDecoderModule):

    def __init__(
        self,
        instances: Dict[str, Any],
        single_tooth: Dict[str, Any],
        batch_size: int,
        x_axis_flip: bool,
        score_thr: float,
        out_dir: Path,
        **kwargs,
    ) -> None:
        super().__init__(out_dir=out_dir, **kwargs)
        
        # stage 1
        self.instance_model = nn.ERFNet(
            in_channels=1,
            out_channels=instances['out_channels'],
            num_filters=instances['num_filters'],
        )
        self.load_ckpt(self.instance_model, instances)

        self.identify_model = nn.Identification(
            num_features=instances['num_filters'],
            out_channels=32,
        )
        self.load_ckpt(self.identify_model, instances)

        # stage 2
        self.single_tooth_model = nn.UNet(out_channels=single_tooth['out_channels'])
        self.load_ckpt(self.single_tooth_model, single_tooth)

        self.gaussian = torch.from_numpy(self.gaussian_kernel())[None, None]
        self.gaussian_down = torch.from_numpy(self.gaussian_kernel(factor=0.5))[None, None]
        self.batch_size = batch_size
        self.x_axis_flip = x_axis_flip
        self.score_thr = score_thr
        self.voxel_size = instances['voxel_size']
        self.min_seed_score = instances['min_seed_score']
        self.min_cluster_size = instances['min_cluster_size']
        self.min_unclustered = instances['min_unclustered']
        self.only_dentalnet = out_dir.name == 'dentalnetPr'

    def instances_stage(
        self,
        x_up: TensorType[1, 1, 'D', 'H', 'W', torch.float32],
    ) -> TensorType[1, 1, 'D', 'H', 'W', torch.int64]:
        # interpolate input to low resolution
        out_size = tuple([int(0.5 * dim) for dim in x_up.shape[-3:]])
        x = torch.nn.functional.interpolate(
            input=x_up,
            size=out_size,
            mode='trilinear',
        )

        # run model with sliding window and gaussian-weighted aggregation
        offsets, sigmas = torch.zeros(2, 1, 3, *x.shape[-3:]).to(x)
        seeds = torch.zeros(1, 1, *x.shape[-3:]).to(x)
        features, features_flip = torch.zeros(
            2, 1, self.identify_model.num_features,
            *[int(np.ceil(dim / 2)) for dim in x.shape[-3:]],
        ).to(x)
        n = torch.zeros_like(seeds)
        n_down = torch.zeros_like(features[:, :1])
        for slices, slices_down in zip(
            self.crop_slices(x), self.crop_slices(x, factor=0.5),
        ):
            out = self.instance_model(x[slices])
            if self.x_axis_flip:
                out_flip = self.instance_model(torch.flip(x[slices], (2,)))
                out_flip = tuple([torch.flip(of, (2,)) for of in out_flip])

                # TODO: how to deal with learned center of attraction?
                out[0][:, 3:] = torch.mean(torch.stack((out[0][:, 3:], out_flip[0][:, 3:])), 0)
                out[1] = torch.mean(torch.stack((out[1], out_flip[1])), 0)
                features_flip[slices_down] += out_flip[2] * self.gaussian_down.to(x)

            offsets[slices] += out[0][:, :3] * self.gaussian.to(x)
            sigmas[slices] += out[0][:, 3:] * self.gaussian.to(x)
            seeds[slices] += out[1] * self.gaussian.to(x)
            features[slices_down] += out[2] * self.gaussian_down.to(x)
            n[slices] += self.gaussian.to(x)
            n_down[slices_down] += self.gaussian_down.to(x)

        offsets /= n
        sigmas /= n
        seeds /= n
        features /= n_down
        features_flip /= n_down

        # determine instances by clustering
        instances = learned_region_cluster(
            offsets, sigmas, seeds,
            voxel_size=self.voxel_size,
            min_seed_score=self.min_seed_score,
            min_cluster_size=self.min_cluster_size,
            min_unclustered=self.min_unclustered,
        )

        # identify tooth classes of instances
        _, logits = self.identify_model(features, instances)
        if self.x_axis_flip:
            _, logits_flip = self.identify_model(features_flip, instances)
            logits[:, :8] += logits_flip[:, 8:16]
            logits[:, 8:16] += logits_flip[:, :8]
            logits[:, 16:24] += logits_flip[:, 24:]
            logits[:, 24:] += logits_flip[:, 16:24]
        class_probs = logits.softmax(-1)
        classes = torch.where(class_probs.amax(-1) < self.score_thr, 0, logits.argmax(-1) + 1)
        instances[instances > 0] = classes[instances[instances > 0] - 1]

        # interpolate output back to original resolution
        out = torch.zeros(x_up.shape).to(instances)
        for label in torch.unique(instances)[1:]:
            mask = torch.nn.functional.interpolate(
                input=(instances == label)[None].float(),
                size=x_up.shape[-3:],
                mode='trilinear',
            ) >= 0.5
            out[mask] = label

        return out
    
    def single_tooth_stage(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        instances: TensorType[1, 1, 'D', 'H', 'W', torch.int64],
    ) -> TensorType[1, 1, 'D', 'H', 'W', torch.int64]:
        patches, patch_slices = self.tooth_instances(x, instances[0])
        patches = patches[:, :1]

        # predict single tooth segmentation in patch batches
        seg = torch.zeros_like(patches[:0])
        for start in range(0, patches.shape[0], self.batch_size):
            patches_batch = patches[start:start + self.batch_size]
            seg = torch.cat((seg, self.single_tooth_model(patches_batch)))
        if self.x_axis_flip:
            for start in range(0, patches.shape[0], self.batch_size):
                patches_batch = torch.flip(patches[start:start + self.batch_size], (2,))
                seg_flip = torch.flip(self.single_tooth_model(patches_batch), (2,))
                seg[start:start + self.batch_size] += seg_flip
        
        # put tooth segmentations back in original volume with corresponding FDI
        seg_logits = torch.zeros_like(x[0, 0])
        out = torch.zeros_like(instances[0, 0])
        for patch_seg, label, slices in zip(
            seg, torch.unique(instances)[1:], patch_slices,
        ):
            out[slices] = torch.where(seg_logits[slices] < patch_seg, label, out[slices])
            seg_logits[slices] = torch.maximum(seg_logits[slices], patch_seg)

        return out[None, None]
    
    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 1, 'D', 'H', 'W', torch.int64]:
        # stage 1
        instances = self.instances_stage(x)
        if not torch.any(instances) or self.only_dentalnet:
            return instances
        
        # stage 2
        instances = self.single_tooth_stage(x, instances)

        return instances
    
    
    def predict_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Path,
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ):
        features, scan_file, affine, shape = batch

        instances = self(features)

        self.save_output(instances, scan_file, affine, shape)
        