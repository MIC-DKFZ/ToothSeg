from hashlib import blake2s
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from numpy.typing import NDArray
from torch.utils.data import Dataset

from baselines.data.utils.cache import DatasetCache


class VolumeDataset(Dataset):
    """Dataset to load 3D volumes with intensity values."""

    def __init__(
        self,
        stage: str,
        root: Union[str, Path],
        cache_dir: Union[str, Path],
        files: List[Tuple[Path, ...]],
        pre_transform: Callable[..., Dict[str, Any]]=dict,
        transform: Callable[..., Dict[str, Any]]=dict,
    ) -> None:
        super().__init__()

        self.stage = stage
        self.root = Path(root)
        self.files = files
        self.pre_transform = pre_transform
        self.transform = transform
        self.cache = DatasetCache(
            files=self.files,
            cache_path=root / cache_dir / str(hash(self)),
            disable=stage == 'predict',
        )

    def load_inputs(self, *files: Tuple[Path, ...]):
        raise NotImplementedError

    def load_targets(self, file: Path):
        raise NotImplementedError

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Union[NDArray[Any], int]]:
        # load data from cache or storage
        if index in self.cache:
            data_dict = self.cache[index]
        else:
            if self.stage == 'predict':
                input_files = self.files[index]
                data_dict = self.load_inputs(*input_files)
            else:
                input_file = self.files[index][0]
                target_files = self.files[index][1:]
                data_dict = {
                    **self.load_inputs(input_file),
                    **self.load_targets(*target_files),
                }
            
            # apply preprocessing transformations and cache
            data_dict = self.pre_transform(**data_dict)
            self.cache[index] = data_dict

        # apply data transformations
        data_dict = self.transform(**data_dict)

        return data_dict

    def __hash__(self) -> int:
        h = blake2s()
        h.update(repr(self.files).encode('utf-8'))
        h.update(repr(self.pre_transform).encode('utf-8'))
        return int(h.hexdigest(), base=16)

    def __len__(self) -> int:
        return len(self.files)
