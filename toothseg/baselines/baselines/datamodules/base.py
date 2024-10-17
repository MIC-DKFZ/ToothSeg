from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Dataset

from baselines.data.datasets.teeth import TeethSegDataset
import baselines.data.transforms as T


class VolumeDataModule(pl.LightningDataModule):
    """Implements data module that loads 3D volumes with intensity values."""

    def __init__(
        self,

        root: str,
        scan_dir: str,
        seg_dir: str,
        instances_dir: Union[List[str], str],
        cache_dir: str,

        exclude: List[str],
        regex_filter: str,
        val_size: float,
        test_size: float,
        include_val_as_train: bool,

        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,

        regular_spacing: float,
        norm_clip: Optional[Tuple[int, int]],
        norm_method: str,
        pre_transform: list[str],
        aug_transform: list[str],
        transform: list[str],

        **kwargs: Dict[str, Any],
    ):        
        super().__init__()
        
        self.root = Path(root)
        self.scan_dir = scan_dir
        self.seg_dir = seg_dir
        if not isinstance(instances_dir, list):
            instances_dir = [] if not instances_dir else [instances_dir]
        self.instances_dirs = instances_dir
        self.cache_dir = cache_dir

        self.exclude = '|'.join(exclude) if exclude else '\n'
        self.filter = regex_filter
        self.val_size = val_size
        self.test_size = test_size
        self.include_val = include_val_as_train

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed

        self.cfg = kwargs
        self.pre_transform = T.Compose(
            T.RegularSpacing(spacing=regular_spacing),
            T.NaturalHeadPositionOrient(),
            T.Pad(kwargs['crop_size']),
            self.build_transforms(pre_transform),
        )
        self.aug_transform = aug_transform
        self.transform = transform
        self.default_transforms = T.Compose(
            T.IntensityAsFeatures(norm_clip, norm_method),
            T.ToTensor(),
        )

    def build_transforms(self, transforms: list[str], **kwargs):
        if not transforms:
            return dict
        
        cfg = {**self.cfg, **kwargs}
        transform_list = []
        for transform in transforms:
            transform = getattr(T, transform)(**cfg)
            transform_list.append(transform)

        return T.Compose(*transform_list)

    def _filter_files(
        self,
        pattern: str,
    ) -> List[Path]:
        files = sorted(self.root.glob(pattern))
        files = [f for f in files if re.search(self.filter, str(f))]
        files = [f for f in files if re.search(self.exclude, str(f)) is None]
        files = [f.relative_to(self.root) for f in files]

        return files    
    
    def _files(self, stage: str) -> List[Tuple[Path, ...]]:
        scan_files = self._filter_files(f'{self.scan_dir}/*.nii.gz')

        if stage == 'predict':
            return list(zip(scan_files))

        seg_files = self._filter_files(f'{self.seg_dir}/*.nii.gz')

        if not self.instances_dirs:
            return list(zip(scan_files, seg_files))
        
        instances_files = []
        for instances_dir in self.instances_dirs:
            files = self._filter_files(f'{instances_dir}/*.nii.gz')
            instances_files.append(files)

        print('Total number of files:', len(seg_files))

        return list(zip(scan_files, seg_files, *instances_files))

    def _split(
        self,
        files: List[Tuple[Path, ...]],
    ) -> Tuple[
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
    ]:
        val_files = len(files) * self.val_size
        test_files = len(files) * self.test_size
        if val_files < 1 and test_files < 1:
            return files, [], []
        elif val_files > len(files) - 1:
            return [], files, []
        elif test_files > len(files) - 1:
            return [], [], files
    
        ss = ShuffleSplit(
            n_splits=1,
            test_size=self.val_size + self.test_size,
            random_state=self.seed,
        )
        train_idxs, val_test_idxs = next(ss.split(files))

        train_files = [files[i] for i in train_idxs]
        val_test_files = [files[i] for i in val_test_idxs]

        if val_files > len(val_test_files) - 1:
            return train_files, val_test_files, [],
        elif test_files > len(val_test_files) - 1:
            return train_files, [], val_test_files
    
        ss = ShuffleSplit(
            n_splits=1,
            test_size=self.test_size / (self.val_size + self.test_size),
            random_state=self.seed,
        )
        val_idxs, test_idxs = next(ss.split(val_test_files))

        val_files = [val_test_files[i] for i in val_idxs]
        test_files = [val_test_files[i] for i in test_idxs]

        return train_files, val_files, test_files
    
    def setup(self, stage: Optional[str]=None) -> None:
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files, _ = self._split(files)

            rng = np.random.default_rng(self.seed)
            train_transform = T.Compose(
                self.build_transforms(self.aug_transform, rng=rng),
                self.build_transforms(self.transform, rng=rng),
                self.default_transforms,
            )
            val_transform = T.Compose(
                self.build_transforms(self.transform, rng=rng),
                self.default_transforms,
            )

            self.train_dataset = TeethSegDataset(
                stage='fit',
                root=self.root,
                cache_dir=self.cache_dir,
                files=train_files + (val_files if self.include_val else []),
                pre_transform=self.pre_transform,
                transform=train_transform,
            )
            self.val_dataset = TeethSegDataset(
                stage='fit',
                root=self.root,
                cache_dir=self.cache_dir,
                files=val_files,
                pre_transform=self.pre_transform,
                transform=val_transform,
            )

        if stage is None or stage == 'test':
            files = self._files('test')
            train_files, val_files, _ = self._split(files)
            
            test_transform = T.Compose(
                self.build_transforms(self.transform),
                self.default_transforms,
            )

            self.test_dataset = TeethSegDataset(
                stage='test',
                root=self.root,
                cache_dir=self.cache_dir,
                files=train_files + val_files,
                pre_transform=self.pre_transform,
                transform=test_transform,
            )

        if stage is None or stage == 'predict':
            all_files = self._files('predict')

            self.predict_dataset = TeethSegDataset(
                stage='predict',
                root=self.root,
                cache_dir=self.cache_dir,
                files=all_files,
                pre_transform=self.pre_transform,
                transform=self.default_transforms,
            )

    def _dataloader(
        self,
        dataset: Dataset,
        collate_fn: Callable[..., Dict[str, Any]],
        shuffle: bool=False,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.train_dataset, self.fit_collate_fn, shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, self.fit_collate_fn)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(self.predict_dataset, self.predict_collate_fn)
