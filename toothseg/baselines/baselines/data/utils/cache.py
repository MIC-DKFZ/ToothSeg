import h5py
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm


class DatasetCache(dict):
    """Implements cache to load and store preprocessed dataset from storage."""

    def __init__(
        self,
        files: List[Tuple[Path, ...]],
        cache_path: Path,
        disable: bool=False,
    ) -> None:
        super().__init__()

        if cache_path.exists():
            print(f'Loading pre-processed samples from {cache_path}.')
            for path in tqdm(sorted(cache_path.glob('*.h5'))):
                try:
                    with h5py.File(path, 'r') as f:
                        super().__setitem__(int(path.stem), None)
                except Exception:
                    print(path, 'corrupted!')
                    path.unlink()

        if not disable:
            cache_path.mkdir(parents=True, exist_ok=True)

        self.files = files
        self.cache_path = cache_path
        self.disable = disable

    def __contains__(self, key: int) -> bool:
        if super().__contains__(key):
            return True
        
        if (self.cache_path / f'{key}.h5').exists():
            super().__setitem__(key, None)
            return True
        
        return False
    
    def __getitem__(self, key: int) -> Any:
        with h5py.File(self.cache_path / f'{key}.h5', 'r') as f:
            data_dict = {}
            for k, v in f.items():
                if k.endswith('_scalar'):
                    k = k[:-7]
                    v = v[0]
                data_dict[k] = v[()]

        data_dict.update({
            'scan_file': self.files[key][0].as_posix(),
            'seg_file': self.files[key][1].as_posix(),
        })

        return data_dict

    def __setitem__(self, key: int, value: Any) -> None:
        if self.disable:
            return
        
        with h5py.File(self.cache_path / f'{key}.h5', 'w') as f:
            for k, v in value.items():
                if isinstance(v, np.generic):
                    k += '_scalar'
                    v = np.array([v])
                if isinstance(v, np.ndarray):
                    f.create_dataset(name=k, data=v, compression='gzip')

        super().__setitem__(key, None)

        if len(self) == len(self.files):
            print('Complete dataset pre-processed!')
