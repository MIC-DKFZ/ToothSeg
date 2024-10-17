from typing import Any, List, Tuple, Union

import torch
from torchmetrics.classification import (
    BinaryF1Score,    
    BinaryJaccardIndex,
    MulticlassF1Score,
)
from torchtyping import TensorType

from baselines.models.base import EncoderDecoderModule
from baselines.models.cluster import learned_region_cluster
import baselines.nn as nn


class DentalNet(EncoderDecoderModule):

    def __init__(
        self,
        out_channels: Tuple[int, int, None],
        crop_size: Union[Tuple[int, int, int], int],
        voxel_size: float,
        num_filters: int,
        loss: dict[str, Any],
        min_seed_score: float,
        min_cluster_size: int,
        min_unclustered: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.instance_model = nn.ERFNet(
            in_channels=1,
            out_channels=out_channels,
            num_filters=num_filters,
        )
        with torch.no_grad():
            output_conv = self.instance_model.decoders[0].output_conv
            # offsets
            output_conv.weight[:, :3].fill_(0)
            output_conv.bias[:3].fill_(0)
            # sigmas
            output_conv.weight[:, 3:].fill_(0)
            output_conv.bias[3:].fill_(1)

        self.identify_model = nn.Identification(
            num_features=num_filters,
            out_channels=32,
        )

        self.instance_criterion = nn.SpatialEmbeddingLoss(
            crop_size, voxel_size, **loss['spatial_embedding'],
        )
        self.identify_criterion = nn.IdentificationLoss(**loss['identify'])

        self.iou = BinaryJaccardIndex()
        self.dice = BinaryF1Score()
        self.f1 = MulticlassF1Score(num_classes=32)

        self.voxel_size = voxel_size
        self.min_seed_score = min_seed_score
        self.min_cluster_size = min_cluster_size
        self.min_unclustered = min_unclustered

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        labels: TensorType['B', 'D', 'H', 'W', torch.int64],
    ) -> Tuple[
        TensorType['B', 3, 'D', 'H', 'W', torch.float32],
        TensorType['B', '1|3', 'D', 'H', 'W', torch.float32],
        TensorType['B', 1, 'D', 'H', 'W', torch.float32],
        List[TensorType['F', 'N', torch.float32]],
        TensorType['K', 32, torch.float32],
    ]:
        spatial_embeds, seeds, features = self.instance_model(x)
        prototypes, classes = self.identify_model(features, labels)

        offsets, sigmas = spatial_embeds.split(3, dim=1)

        return offsets, sigmas, seeds, prototypes, classes

    def training_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            TensorType['B', 'D', 'H', 'W', torch.int64],
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, labels = batch

        offsets, sigmas, seeds, prototypes, classes = self(x, labels)

        instance_loss = self.instance_criterion(offsets, sigmas, seeds, labels)
        identify_loss = self.identify_criterion(prototypes, classes, labels)

        loss = instance_loss + identify_loss
        
        self.log_dict({
            'loss/train_instance': instance_loss,
            'loss/train_identify': identify_loss,
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
    ) -> TensorType[torch.float32]:
        x, labels = batch

        offsets, sigmas, seeds, prototypes, classes = self(x, labels)

        instance_loss = self.instance_criterion(offsets, sigmas, seeds, labels)
        identify_loss = self.identify_criterion(prototypes, classes, labels)

        loss = instance_loss + identify_loss

        log_dict = {
            'loss/val_instance': instance_loss,
            'loss/val_identify': identify_loss,
            'loss/val': loss,
        }

        if self.trainer.state.fn == 'validate' or self.current_epoch >= 5:
            instances = learned_region_cluster(
                offsets, sigmas, seeds,
                voxel_size=self.voxel_size,
                min_seed_score=self.min_seed_score,
                min_cluster_size=self.min_cluster_size,
                min_unclustered=self.min_unclustered,
            )
            self.dice((instances > 0).long(), (labels > 0).long())
            self.iou((instances > 0).long(), (labels > 0).long())

            log_dict.update({
                'dice/val': self.dice,
                'iou/val': self.iou,
            })
        
        if labels.amax() > 0:
            target = torch.cat([
                target.unique()[1:] - 1 for target in labels
            ])
            self.f1(classes, target)
            log_dict.update({'f1/val': self.f1})

        self.log_dict(log_dict)
