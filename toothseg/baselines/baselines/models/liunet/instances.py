from typing import Tuple

import torch
from torchmetrics.classification import (
    BinaryF1Score,    
    BinaryJaccardIndex,
    MultilabelF1Score,
)
from torchtyping import TensorType

from baselines.models.base import EncoderDecoderModule
import baselines.nn as nn


class LiuToothInstanceSegmentationNet(EncoderDecoderModule):

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.model = nn.SwinUNETR(
            img_size=kwargs['crop_size'],
            in_channels=2,
            out_channels=33,
        )
        self.seg_criterion = nn.SegmentationLoss(
            ce_weight=1.0, dice_weight=1.0,
        )

        self.iou = BinaryJaccardIndex()
        self.dice = BinaryF1Score()
        self.f1 = MultilabelF1Score(num_labels=33)

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],     
    ) -> TensorType['B', 33, 'D', 'H', 'W', torch.float32]:
        seg = self.model(x)

        return seg

    def training_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            TensorType['B', 'D', 'H', 'W', torch.int64],
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, labels = batch

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
            TensorType['B', 'D', 'H', 'W', torch.int64],
        ],
        batch_idx: int,
    ) -> None:
        x, labels = batch

        seg = self(x)

        loss = self.seg_criterion(seg, labels)

        pred = seg.argmax(1)
        self.dice((pred > 0).long(), (labels > 0).long())
        self.iou((pred > 0).long(), (labels > 0).long())
        
        pred_multilabels = torch.zeros((x.shape[0], 33)).to(pred)
        multilabels = torch.zeros((x.shape[0], 33)).to(pred)
        for i, (pred, label) in enumerate(zip(pred, labels)):
            pred_multilabels[i][torch.unique(pred)] = 1
            multilabels[i][torch.unique(label)] = 1
        self.f1(pred_multilabels, multilabels)

        self.log_dict({
            'loss/val': loss,
            'dice/val': self.dice,
            'iou/val': self.iou,
            'f1/val': self.f1,
        })
