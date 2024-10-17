from typing import Tuple

import torch
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
from torchtyping import TensorType

from baselines.models.base import EncoderDecoderModule
import baselines.nn as nn


class SingleToothSegmentationNet(EncoderDecoderModule):

    def __init__(
        self,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.model = nn.UNet(out_channels=out_channels)
        self.seg_criterion = nn.BinarySegmentationLoss()

        self.iou = BinaryJaccardIndex()
        self.dice = BinaryF1Score()

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],     
    ) -> TensorType['B', 1, 'D', 'H', 'W', torch.float32]:
        seg = self.model(x)
        
        return seg

    def training_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            TensorType['B', 1, 'D', 'H', 'W', torch.bool]
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, labels = batch
        if x.shape[0] == 0:
            return

        seg = self(x)

        loss = self.seg_criterion(seg, labels)      

        self.log_dict({
            'loss/train': loss,
        })

        return loss

    def validation_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            TensorType['B', 1, 'D', 'H', 'W', torch.bool]
        ],
        batch_idx: int,
    ) -> None:
        x, labels = batch
        if x.shape[0] == 0:
            return

        seg = self(x)

        loss = self.seg_criterion(seg, labels)

        self.dice(seg[:, 0], labels.long())
        self.iou(seg[:, 0], labels.long())

        self.log_dict({
            'loss/val': loss,
            'dice/val': self.dice,
            'iou/val': self.iou,
        })
