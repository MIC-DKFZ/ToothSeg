from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
from scipy import ndimage
import torch
from torchtyping import TensorType
from baselines.models.base import EncoderDecoderModule
import baselines.nn as nn


class CuiNet(EncoderDecoderModule):

    def __init__(
        self,
        roi: Dict[str, Any],
        centroids: Dict[str, Any],
        skeletons: Dict[str, Any],
        single_tooth: Dict[str, Any],
        x_axis_flip: bool,
        score_thr: float,
        return_type: Literal['fdi', 'iso', 'instances'],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # stage 1
        self.roi_model = nn.VNet(**roi['architecture'])
        self.load_ckpt(self.roi_model, roi)

        # stage 2a + 2b
        self.centroids_model = nn.VNet(**centroids['architecture'])
        self.load_ckpt(self.centroids_model, centroids)
        self.skeletons_model = nn.VNet(**skeletons['architecture'])
        self.load_ckpt(self.skeletons_model, skeletons)

        # stage 3
        self.single_tooth_model = nn.VNet(**single_tooth['architecture'])
        self.load_ckpt(self.single_tooth_model, single_tooth)

        self.gaussian = torch.from_numpy(self.gaussian_kernel())[None, None]
        self.x_axis_flip = x_axis_flip
        self.score_thr = score_thr
        self.return_type = return_type
    
    def _fit_bbox(
        self,
        mask,
    ):
        if np.any(mask):
            centroid = np.column_stack(np.nonzero(mask)).mean(0).astype(int)
            keypoint = np.zeros_like(mask)
            keypoint[centroid[0], centroid[1], centroid[2]] = True
            slices = ndimage.find_objects(keypoint)[0]
        else:
            slices = [slice(0, dim) for dim in mask.shape]
        
        # crop or expand bounding box to exactly crop_size
        out = ()
        for slc, dim, crop_size in zip(slices, mask.shape, self.crop_size):
            diff = crop_size - (slc.stop - slc.start)
            diff = diff // 2, diff // 2 + diff % 2
            slc = slice(slc.start - diff[0], slc.stop + diff[1])
            diff = dim - min(slc.start, 0) - max(dim, slc.stop)
            slc = slice(slc.start + diff, slc.stop + diff)
            out += (slc,)
            
        return (slice(None), slice(None)) + out
    
    def roi_stage(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 'C', 'D', 'H', 'W', torch.bool]:
        # run model with sliding window and gaussian-weighted aggregation
        out = torch.zeros_like(x)
        n = torch.zeros_like(x)
        for slices in self.crop_slices(x):
            seg_crop = self.roi_model(x[slices])
            if self.x_axis_flip:
                seg_crop_flip = self.roi_model(torch.flip(x[slices], (2,)))
                seg_crop_flip = torch.flip(seg_crop_flip, (2,))

                tp = ((seg_crop > 0) & (seg_crop_flip > 0)).sum()
                fp = ((seg_crop <= 0) & (seg_crop_flip > 0)).sum()
                fn = ((seg_crop > 0) & (seg_crop_flip <= 0)).sum()
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
                if f1 < 0.8:
                    n[slices] += self.gaussian.to(x)
                    continue
                
                seg_crop += seg_crop_flip

            out[slices] += torch.sigmoid(seg_crop) * self.gaussian.to(x)
            n[slices] += self.gaussian.to(x)
        out /= n

        # determine binary segmentation
        return out >= 0.5
    
    def detect_stage(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
    ) -> TensorType[1, 3, 'D', 'H', 'W', torch.int64]:
        seg_c, offsets_c = self.centroids_model(x)
        seg_s, offsets_s = self.skeletons_model(x)

        if self.x_axis_flip:
            seg_c_flip, offsets_c_flip = self.centroids_model(torch.flip(x, (2,)))
            seg_c_flip, offsets_c_flip = torch.flip(seg_c_flip, (2,)), torch.flip(offsets_c_flip, (2,))
            offsets_c_flip[:, 0] *= -1
            offsets_c = torch.where(seg_c_flip > seg_c, offsets_c_flip, offsets_c)
            seg_c = torch.maximum(seg_c, seg_c_flip)

            seg_s_flip, offsets_s_flip = self.skeletons_model(torch.flip(x, (2,)))
            seg_s_flip, offsets_s_flip = torch.flip(seg_s_flip, (2,)), torch.flip(offsets_s_flip, (2,))
            offsets_s_flip[:, 0] *= -1
            offsets_s = torch.where(seg_s_flip > seg_s, offsets_s_flip, offsets_s)
            seg_s = torch.maximum(seg_s, seg_s_flip)

        seg = (seg_c >= 0) & (seg_s >= 0)
        cluster_idxs = self.cluster(seg, offsets_c)
        if cluster_idxs.shape[0] == 0 or not torch.any(cluster_idxs):
            return torch.zeros_like(x[:, :1], dtype=torch.int64).tile(1, 3, 1, 1, 1)

        masks = self.tooth_representations(seg, torch.zeros_like(offsets_c), cluster_idxs, min_density=1)
        centroids = self.tooth_representations(seg, offsets_c, cluster_idxs)
        skeletons = self.tooth_representations(seg, offsets_s, cluster_idxs)

        instances = torch.stack((masks, centroids, skeletons))[None]

        return instances
    
    def single_tooth_stage(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        instances: TensorType[1, 'I', 'D', 'H', 'W', torch.int64],
    ) -> TensorType['D', 'H', 'W', torch.int64]:
        patches, patch_slices = self.tooth_instances(x, instances[0, 1:])
        
        seg, _, _, class_logits = self.single_tooth_model(patches)
        seg = seg[:, 0]

        if self.x_axis_flip:
            seg_flip, _, _, class_logits_flip = self.single_tooth_model(
                torch.flip(patches, (2,)),
            )
            seg += torch.flip(seg_flip[:, 0], (1,))
            class_logits[:, :8] += class_logits_flip[:, 8:16]
            class_logits[:, 8:16] += class_logits_flip[:, :8]
            class_logits[:, 16:24] += class_logits_flip[:, 24:]
            class_logits[:, 24:] += class_logits_flip[:, 16:24]

        seg_logits = torch.zeros_like(x[0, 0])
        out = torch.zeros_like(x[0, 0], dtype=torch.int64)
        for patch_seg, logits, slices in zip(seg, class_logits, patch_slices):
            if torch.softmax(logits, dim=0).amax() < self.score_thr:
                continue
            
            label = logits.argmax() + 1

            out[slices] = torch.where(seg_logits[slices] < patch_seg, label, out[slices])
            seg_logits[slices] = torch.maximum(seg_logits[slices], patch_seg)

        return out[None, None]

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
    ) -> TensorType[1, 1, 'D', 'H', 'W', torch.int64]:
        # stage 1
        tooth_mask = self.roi_stage(x)
        if not torch.any(tooth_mask) or self.return_type == 'binary':
            if self.return_type == 'instances':
                tooth_mask = torch.tile(tooth_mask, (1, 3, 1, 1, 1))
                
            return tooth_mask.to(torch.int64)

        # determine fixed-size slices around teeth
        roi_slices = self._fit_bbox(tooth_mask[0, 0].cpu().numpy())
        roi = torch.cat((x[roi_slices], tooth_mask[roi_slices]), dim=1)
        
        # stage 2
        instances = self.detect_stage(roi)        
        if not torch.any(instances) or self.return_type == 'instances':
            out = torch.zeros(*instances.shape[:2], *x.shape[-3:]).to(instances)
            out[roi_slices] = instances
            return out
        
        # stage 3
        instances = self.single_tooth_stage(roi[:, :1], instances)
        out = torch.zeros_like(x, dtype=torch.int64)
        out[roi_slices] = instances
        
        return out
    
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
