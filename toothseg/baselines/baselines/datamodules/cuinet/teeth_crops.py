from typing import Any, Dict, List, Tuple
import torch
from torchtyping import TensorType

from baselines.datamodules.base import VolumeDataModule


class CuiNetToothSkeletonsCropSegDataModule(VolumeDataModule):

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        Tuple[
            TensorType['B', 'D', 'H', 'W', torch.bool],
            TensorType['B', 3, 'D', 'H', 'W', torch.float32],
        ],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.stack(batch_dict['features'])
        instances = torch.stack(batch_dict['instances'])
        labels = torch.stack(batch_dict['labels']) > 0
        offsets = torch.stack(batch_dict['offsets'])

        inputs = torch.cat((features, instances), dim=1)
        
        return inputs, (labels, offsets)
