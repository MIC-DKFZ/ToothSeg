from typing import Any, Dict, List, Tuple
import torch
from torchtyping import TensorType
from baselines.datamodules.base import VolumeDataModule


class CuiNetToothPatchKeypointSegDataModule(VolumeDataModule):

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        Tuple[
            TensorType['B', 'D', 'H', 'W', torch.bool],
            TensorType['B', 'D', 'H', 'W', torch.float32],
            TensorType['B', 'D', 'H', 'W', torch.float32],
            TensorType['B', torch.int64],
        ]
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.cat(batch_dict['features'])
        instances = torch.cat(batch_dict['instances'])

        inputs = torch.cat((features, instances[:, 1:]), axis=1)
        labels = torch.cat(batch_dict['labels'])
        boundaries = torch.cat(batch_dict['boundaries'])
        keypoints = torch.cat(batch_dict['keypoints'])
        classes = torch.cat(batch_dict['unique_labels'])

        return inputs, (labels, boundaries, keypoints, classes)
