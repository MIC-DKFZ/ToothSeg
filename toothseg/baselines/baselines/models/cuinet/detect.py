from typing import Any, Dict, Tuple

import torch
from torchmetrics.classification import (
    BinaryF1Score,    
    BinaryJaccardIndex,
)
from torchtyping import TensorType

from baselines.models.base import EncoderDecoderModule
import baselines.nn as nn


class ToothSegmentationOffsetsNet(EncoderDecoderModule):

    def __init__(
        self,
        architecture: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.model = nn.VNet(**architecture)
        self.seg_criterion = nn.BinarySegmentationLoss()
        self.offset_criterion = nn.SmoothL1Loss()

        self.iou = BinaryJaccardIndex()
        self.dice = BinaryF1Score()

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],     
    ) -> Tuple[
        TensorType['B', 1, 'D', 'H', 'W', torch.float32],
        TensorType['B', 3, 'D', 'H', 'W', torch.float32],
    ]:
        seg, offsets = self.model(x)

        return seg, offsets

    def training_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Tuple[
                TensorType['B', 'D', 'H', 'W', torch.bool],
                TensorType['B', 3, 'D', 'H', 'W', torch.float32],
            ],
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, (labels, offsets) = batch

        seg, pred_offsets = self(x)

        seg_loss = self.seg_criterion(seg, labels)
        mask = torch.tile(labels[:, None], (1, 3, 1, 1, 1))
        offset_loss = self.offset_criterion(pred_offsets[mask], offsets[mask])
        loss = seg_loss + offset_loss

        self.log_dict({
            'loss/train_seg': seg_loss,
            'loss/train_offset': offset_loss,
            'loss/train': loss,
        })

        return loss

    def validation_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Tuple[
                TensorType['B', 'D', 'H', 'W', torch.bool],
                TensorType['B', 3, 'D', 'H', 'W', torch.float32],
            ],
        ],
        batch_idx: int,
    ) -> None:
        x, (labels, offsets) = batch

        seg, pred_offsets = self(x)

        seg_loss = self.seg_criterion(seg, labels)
        mask = torch.tile(labels[:, None], (1, 3, 1, 1, 1))
        offset_loss = self.offset_criterion(pred_offsets[mask], offsets[mask])
        loss = seg_loss + offset_loss

        self.dice(seg[:, 0], labels.long())
        self.iou(seg[:, 0], labels.long())

        self.log_dict({
            'loss/val_seg': seg_loss,
            'loss/val_offset': offset_loss,
            'loss/val': loss,
            'dice/val': self.dice,
            'iou/val': self.iou,
        })
