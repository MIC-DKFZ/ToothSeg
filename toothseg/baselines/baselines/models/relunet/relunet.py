from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy import ndimage
import torch
from torchtyping import TensorType

from baselines.models.base import EncoderDecoderModule
import baselines.nn as nn


class ReluNet(EncoderDecoderModule):

    def __init__(
        self,
        roi: Dict[str, Any],
        multiclass: Dict[str, Any],
        single_tooth: Dict[str, Any],
        batch_size: int,
        x_axis_flip: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # stage 1
        self.roi_model = nn.UNet(out_channels=roi['out_channels'])
        self.load_ckpt(self.roi_model, roi)

        # stage 2
        self.multiclass_model = nn.UNet(out_channels=multiclass['out_channels'])
        self.load_ckpt(self.multiclass_model, multiclass)

        # stage 3
        self.single_tooth_model = nn.UNet(out_channels=single_tooth['out_channels'])
        self.load_ckpt(self.single_tooth_model, single_tooth)

        self.batch_size = batch_size
        self.x_axis_flip = x_axis_flip
    
    def _fit_bbox(
        self,
        mask,
        crop_size=None,
    ):
        slices = ndimage.find_objects(mask)[0]

        crop_size = self.crop_size if crop_size is None else crop_size
        
        # crop or expand bounding box to exactly crop_size
        out = ()
        for slc, dim, crop_size in zip(slices, mask.shape, crop_size):
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
        # downsample CBCT scan to fixed size
        x_down = torch.nn.functional.interpolate(
            input=x,
            size=self.crop_size,
            mode='trilinear'
        )

        # predict a binary segmentation of teeth
        seg_down = self.roi_model(x_down)
        if self.x_axis_flip:
            seg_down_flip = self.roi_model(torch.flip(x_down, (2,)))
            seg_down += torch.flip(seg_down_flip, (2,))

        # upsample prediction back to original size
        seg = torch.nn.functional.interpolate(
            input=seg_down,
            size=tuple(x.shape[-3:]),
            mode='trilinear'
        )

        # determine binary mask
        out = torch.sigmoid(seg) >= 0.5

        return out
    
    def multiclass_stage(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        tooth_mask: TensorType[1, 1, 'D', 'H', 'W', torch.bool],
        spacing: TensorType[3, torch.float32],
    ) -> TensorType['B', 'C', 'D', 'H', 'W', torch.int64]:
        up_shape = 0.7 / spacing.cpu() * torch.tensor(self.crop_size)
        up_shape = tuple(up_shape.ceil().long().tolist())

        x_pad, pads = self.pad(x, up_shape)
        tooth_mask_pad, pads = self.pad(tooth_mask, up_shape)
        up_slices = self._fit_bbox(tooth_mask_pad[0, 0].cpu().numpy(), up_shape)


        # downsample volumes to fixed spacing
        x_down = torch.nn.functional.interpolate(
            input=x_pad[up_slices],
            size=self.crop_size,
            mode='trilinear'
        )

        # predict a multiclass segmentation of teeth
        seg_down = torch.zeros((1, 33, *x_down.shape[-3:])).to(x_down)
        seg_down = self.multiclass_model(x_down)
        if self.x_axis_flip:
            seg_down_flip = self.multiclass_model(torch.flip(x_down, (2,)))
            seg_down_flip = torch.flip(seg_down_flip, (2,))
            seg_down[:, 0] += seg_down_flip[:, 0]
            seg_down[:, 1:9] += seg_down_flip[:, 9:17]
            seg_down[:, 9:17] += seg_down_flip[:, 1:9]
            seg_down[:, 17:25] += seg_down_flip[:, 25:]
            seg_down[:, 25:] += seg_down_flip[:, 17:25]
        seg_down = seg_down.argmax(1, keepdim=True)

        # determine multiclass labels of original volume
        out = torch.zeros_like(x).to(torch.int64)
        down_slices = [
            slice(pad[0], dim - pad[1]) for dim, pad in
            zip(up_shape, pads)
        ]
        out[up_slices] = torch.nn.functional.interpolate(
            input=seg_down.float(),
            size=up_shape,
            mode='nearest',
        )[:, :, down_slices[0], down_slices[1], down_slices[2]]

        return out
    
    def single_tooth_stage(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        instances: TensorType[1, 1, 'D', 'H', 'W', torch.int64],
        spacing: TensorType[3, torch.float32],
    ) -> TensorType[1, 1, 'D', 'H', 'W', torch.int64]:
        # determine tooth patches with spacing 0.25mm
        up_shape = 0.25 / spacing.cpu() * torch.tensor(self.patch_size)
        up_shape = tuple(up_shape.ceil().long().tolist())

        self.tooth_patches.patch_size = up_shape
        patches, patch_slices = self.tooth_instances(x, instances[0])
        patches_up = torch.nn.functional.interpolate(
            input=patches[:, :1],
            size=self.patch_size,
            mode='trilinear'
        )

        # predict single tooth segmentation in each patch
        seg_up = torch.zeros_like(patches_up[:0])
        for start in range(0, patches_up.shape[0], self.batch_size):
            patches_batch = patches_up[start:start + self.batch_size]
            seg_up = torch.cat((seg_up, self.single_tooth_model(patches_batch)))
        if self.x_axis_flip:
            for start in range(0, patches_up.shape[0], self.batch_size):
                patches_batch = torch.flip(patches_up[start:start + self.batch_size], (2,))
                seg_up_flip = torch.flip(self.single_tooth_model(patches_batch), (2,))
                seg_up[start:start + self.batch_size] += seg_up_flip

        # interpolate predictions back to original spacing
        seg = torch.nn.functional.interpolate(
            input=seg_up,
            size=tuple(patches.shape[-3:]),
            mode='trilinear'
        )[:, 0].cpu()
        
        # put tooth segmentations back in original volume with corresponding FDI
        seg_logits = torch.zeros_like(x[0, 0], device='cpu')
        out = torch.zeros_like(instances[0, 0], device='cpu')
        for patch_seg, label, slices in zip(
            seg, torch.unique(instances)[1:], patch_slices,
        ):
            # only keep largest connected component
            labels, max_label = ndimage.label(patch_seg >= 0)
            if max_label > 1:
                counts = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, max_label + 1))
                largest = labels == (counts.argmax() + 1)
                patch_seg[~largest] = -10.0

            out[slices] = torch.where(seg_logits[slices] < patch_seg, label.cpu(), out[slices])
            seg_logits[slices] = torch.maximum(seg_logits[slices], patch_seg)

        return out[None, None]

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        spacing: TensorType[3, torch.float32],
    ) -> TensorType[1, 1, 'D', 'H', 'W', torch.int64]:
        # stage 1
        tooth_mask = self.roi_stage(x)
        if not torch.any(tooth_mask):
            return torch.zeros_like(x, dtype=torch.int64)
        
        # stage 2
        instances = self.multiclass_stage(x, tooth_mask, spacing)
        if not torch.any(instances) or self.return_type == 'instances':
            return instances
        
        # stage 3
        instances = self.single_tooth_stage(x, instances, spacing)
        
        return instances
    
    def predict_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Path,
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ):
        features, scan_file, affine, spacing, shape = batch

        instances = self(features, spacing)

        self.save_output(instances, scan_file, affine, shape)
