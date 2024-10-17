from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
from torchtyping import TensorType

from baselines.datamodules.base import VolumeDataModule


class CuiNetFovCropSegDataModule(VolumeDataModule):

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.bool],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.stack(batch_dict['features'])
        labels = torch.stack(batch_dict['labels']) > 0

        return features, labels

    def predict_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        Path,
        TensorType[4, 4, torch.float32],
        TensorType[3, torch.int64],
    ]:
        features = batch[0]['features'][None]
        scan_file = batch[0]['scan_file']
        affine = batch[0]['affine']
        shape = batch[0]['shape']

        return features, scan_file, affine, shape
