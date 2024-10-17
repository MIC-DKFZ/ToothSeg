from typing import Tuple

import baselines.nn as nn
import torch
from torchtyping import TensorType
from torchmetrics.classification import (
    MultilabelF1Score,
    BinaryF1Score,
    BinaryJaccardIndex,
)

from baselines.models.relunet.roi import DownsampledToothSegmentationNet


class MulticlassToothSegmentationNet(DownsampledToothSegmentationNet):

    def __init__(
        self,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__(out_channels=out_channels, **kwargs)

        self.model = nn.UNet(out_channels=out_channels)
        self.seg_criterion = nn.CrossEntropyLoss()

        self.num_classes = out_channels
        self.iou = BinaryJaccardIndex()
        self.dice = BinaryF1Score()
        self.f1 = MultilabelF1Score(num_labels=out_channels)

    def validation_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            TensorType['B', 'D', 'H', 'W', torch.bool],
        ],
        batch_idx: int,
    ) -> None:
        x, labels = batch

        seg = self(x)

        loss = self.seg_criterion(seg, labels)

        preds = seg.argmax(1)
        self.dice((preds > 0).long(), (labels > 0).long())
        self.iou((preds > 0).long(), (labels > 0).long())

        pred_multilabels = torch.zeros((x.shape[0], self.num_classes)).to(preds)
        multilabels = torch.zeros((x.shape[0], self.num_classes)).to(preds)
        for i, (pred, label) in enumerate(zip(preds, labels)):
            pred_multilabels[i][torch.unique(pred)] = 1
            multilabels[i][torch.unique(label)] = 1
        self.f1(pred_multilabels, multilabels)

        self.log_dict({
            'loss/val': loss,
            'dice/val': self.dice,
            'iou/val': self.iou,
            'f1/val': self.f1,
        })
