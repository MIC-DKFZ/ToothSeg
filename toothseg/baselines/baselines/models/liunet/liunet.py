from pathlib import Path
from typing import Tuple, Union

import torch
from torchtyping import TensorType

from baselines.models.base import EncoderDecoderModule
import baselines.nn as nn


class LiuNet(EncoderDecoderModule):

    def __init__(
        self,
        binary: dict,
        instances: dict,
        x_axis_flip: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.binary_model = nn.UNet(out_channels=1)
        self.load_ckpt(self.binary_model, binary)

        self.instance_model = nn.SwinUNETR(
            img_size=self.crop_size,
            in_channels=2,
            out_channels=33,
        )
        self.load_ckpt(self.instance_model, instances)

        self.gaussian = torch.from_numpy(self.gaussian_kernel())[None, None]
        self.x_axis_flip = x_axis_flip

    def binary_stage(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 1, 'D', 'H', 'W', torch.bool]:
        # run model with sliding window and gaussian-weighted aggregation
        pred = torch.zeros_like(x)
        n = torch.zeros_like(x)
        for slices in self.crop_slices(x):
            seg_crop = self.binary_model(x[slices])
            if self.x_axis_flip:
                seg_crop_flip = self.binary_model(torch.flip(x[slices], (2,)))
                seg_crop_flip = torch.flip(seg_crop_flip, (2,))

                tp = ((seg_crop > 0) & (seg_crop_flip > 0)).sum()
                fp = ((seg_crop <= 0) & (seg_crop_flip > 0)).sum()
                fn = ((seg_crop > 0) & (seg_crop_flip <= 0)).sum()
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
                if f1 < 0.8:
                    n[slices] += self.gaussian.to(x)
                    continue
                
                seg_crop += seg_crop_flip

            pred[slices] += torch.sigmoid(seg_crop) * self.gaussian.to(x)
            n[slices] += self.gaussian.to(x)
        pred /= n

        # determine binary mask of teeth
        out = pred >= 0.5

        return out
    
    def instance_stage(
        self,
        x: TensorType['B', 2, 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 1, 'D', 'H', 'W', torch.int64]:
        # run model with sliding window and gaussian-weighted aggregation
        pred = torch.zeros(x.shape[0], 33, *x.shape[-3:]).to(x)
        n = torch.zeros(x.shape[0], 1, *x.shape[-3:]).to(x)
        for slices in self.crop_slices(x):
            seg_crop = self.instance_model(x[slices])
            if self.x_axis_flip:
                seg_crop_flip = self.instance_model(torch.flip(x[slices], (2,)))
                seg_crop_flip = torch.flip(seg_crop_flip, (2,))

                tp = ((seg_crop.argmax(1) > 0) & (seg_crop_flip.argmax(1) > 0)).sum()
                fp = ((seg_crop.argmax(1) <= 0) & (seg_crop_flip.argmax(1) > 0)).sum()
                fn = ((seg_crop.argmax(1) > 0) & (seg_crop_flip.argmax(1) <= 0)).sum()
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
                if f1 < 0.8:
                    n[slices] += self.gaussian.to(x)
                    continue
                
                seg_crop[:, 0] += seg_crop_flip[:, 0]
                seg_crop[:, 1:9] += seg_crop_flip[:, 9:17]
                seg_crop[:, 9:17] += seg_crop_flip[:, 1:9]
                seg_crop[:, 17:25] += seg_crop_flip[:, 25:]
                seg_crop[:, 25:] += seg_crop_flip[:, 17:25]

            pred[slices] += seg_crop * self.gaussian.to(x)
            n[slices] += self.gaussian.to(x)
        pred /= n

        # determine multi-class segmentation of teeth
        out = pred.argmax(1, keepdim=True)

        return out
    
    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> Union[
        TensorType['B', 1, 'D', 'H', 'W', torch.bool],
        TensorType['B', 1, 'D', 'H', 'W', torch.int64],
    ]:
        # stage 1
        binary = self.binary_stage(x)        
        if not torch.any(binary) or self.return_type == 'binary':
            return binary

        # stage 2
        x = torch.cat((x, binary), dim=1)
        instances = self.instance_stage(x)
        
        return instances

    def predict_step(
        self,
        batch: Tuple[
            TensorType['B', 1, 'D', 'H', 'W', torch.float32],
            Path,
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ):
        features, scan_file, affine, shape = batch

        instances = self(features)

        self.save_output(instances, scan_file, affine, shape)
