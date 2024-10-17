from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torchtyping import TensorType

from baselines.datamodules.cuinet.fov_crops import CuiNetFovCropSegDataModule


class ReluNetDownsampledSegDataModule(CuiNetFovCropSegDataModule):
    
    def predict_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        Path,
        TensorType[4, 4, torch.float32],
        TensorType[3, torch.float32],
        TensorType[3, torch.int64],
    ]:
        features = batch[0]['features'][None]
        scan_file = batch[0]['scan_file']
        affine = batch[0]['affine']
        spacing = batch[0]['spacing']
        shape = batch[0]['shape']

        return features, scan_file, affine, spacing, shape
