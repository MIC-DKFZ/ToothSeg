from typing import Any, Dict, Tuple

import torch
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, MulticlassF1Score
from torchtyping import TensorType

from baselines.models.base import EncoderDecoderModule
import baselines.nn as nn


class SingleToothPredictionNet(EncoderDecoderModule):

    def __init__(
        self,
        architecture: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.model = nn.VNet(**architecture)
        self.seg_criterion = nn.BinarySegmentationLoss()
        self.regr_criterion = nn.MSELoss()
        self.class_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.iou = BinaryJaccardIndex()
        self.dice = BinaryF1Score()
        self.f1 = MulticlassF1Score(num_classes=32, ignore_index=-100)

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],     
    ) -> Tuple[
        TensorType['B', 1, 'D', 'H', 'W', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.float32],
        TensorType['B', 32, torch.float32]
    ]:
        seg, boundaries, keypoints, logits = self.model(x)
        
        boundaries = torch.sigmoid(boundaries[:, 0])
        keypoints = torch.sigmoid(keypoints[:, 0])
        
        return seg, boundaries, keypoints, logits

    def training_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Tuple[
                TensorType['B', 'D', 'H', 'W', torch.bool],
                TensorType['B', 'D', 'H', 'W', torch.float32],
                TensorType['B', 'D', 'H', 'W', torch.float32],
                TensorType['B', torch.int64]
            ],
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, (labels, boundaries, keypoints, classes) = batch
        if x.shape[0] <= 1:
            return

        seg, pred_boundaries, pred_keypoints, logits = self(x)

        seg_loss = self.seg_criterion(seg, labels)
        boundary_loss = self.regr_criterion(pred_boundaries, boundaries)
        keypoint_loss = self.regr_criterion(pred_keypoints, keypoints)
        class_loss = self.class_criterion(logits, classes)
        class_loss = torch.tensor(0).to(class_loss) if torch.isnan(class_loss) else class_loss
        
        loss = seg_loss + 0.2 * (boundary_loss + keypoint_loss) + 0.1 * class_loss

        self.log_dict({
            'loss/train_seg': seg_loss,
            'loss/train_boundary': boundary_loss,
            'loss/train_keypoint': keypoint_loss,
            'loss/train_class': class_loss,
            'loss/train': loss,
        })

        return loss

    def validation_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Tuple[
                TensorType['B', 'D', 'H', 'W', torch.bool],
                TensorType['B', 'D', 'H', 'W', torch.float32],
                TensorType['B', 'D', 'H', 'W', torch.float32],
                TensorType['B', torch.int64]
            ],
        ],
        batch_idx: int,
    ) -> None:
        x, (labels, boundaries, keypoints, classes) = batch
        if x.shape[0] <= 1:
            return

        seg, pred_boundaries, pred_keypoints, logits = self(x)

        seg_loss = self.seg_criterion(seg, labels)
        boundary_loss = self.regr_criterion(pred_boundaries, boundaries)
        keypoint_loss = self.regr_criterion(pred_keypoints, keypoints)
        class_loss = self.class_criterion(logits, classes)
        class_loss = torch.tensor(0).to(class_loss) if torch.isnan(class_loss) else class_loss
        
        loss = seg_loss + 0.2 * (boundary_loss + keypoint_loss) + 0.1 * class_loss

        self.dice(seg[:, 0], labels.long())
        self.iou(seg[:, 0], labels.long())
        self.f1(logits, classes)

        self.log_dict({
            'loss/val_seg': seg_loss,
            'loss/val_boundary': boundary_loss,
            'loss/val_keypoint': keypoint_loss,
            'loss/val_class': class_loss,
            'loss/val': loss,
            'dice/val': self.dice,
            'iou/val': self.iou,
            'f1/val': self.f1,
        })
