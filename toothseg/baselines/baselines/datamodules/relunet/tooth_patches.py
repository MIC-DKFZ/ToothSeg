from typing import Any, Dict, List, Tuple
import torch
from torchtyping import TensorType

from baselines.datamodules.base import VolumeDataModule


class ReluNetToothPatchSegDataModule(VolumeDataModule):

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.bool],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.cat(batch_dict['features'])

        labels = torch.cat(batch_dict['labels'])   

        return features, labels
