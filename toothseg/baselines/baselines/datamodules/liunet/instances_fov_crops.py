from typing import Any, Dict, List, Tuple

import torch
from torchtyping import TensorType

from baselines.datamodules.cuinet.fov_crops import CuiNetFovCropSegDataModule


class LiuNetInstancesFovCropSegDataModule(CuiNetFovCropSegDataModule):    

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.int64],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.stack(batch_dict['features'])
        instances = torch.stack(batch_dict['instances'])
        labels = torch.stack(batch_dict['labels'])

        inputs = torch.cat((features, instances > 0), dim=1)

        return inputs, labels
