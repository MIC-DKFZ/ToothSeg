from pathlib import Path
from typing import Any, Dict, Tuple

import nibabel
import numpy as np
from numpy.typing import NDArray

from baselines.data.datasets.base import VolumeDataset


class TeethSegDataset(VolumeDataset):
    """Dataset to load head scans with mandible segmentations."""

    CLASS_MAP = np.concatenate(([0], np.array([
        q * 10 + e
        for q in [1, 2, 3, 4]
        for e in [1, 2, 3, 4, 5, 6, 7, 8]
    ]))).astype(np.uint8)

    def load_inputs(
        self,
        file: Path,
    ) -> Dict[str, NDArray[Any]]:
        img = nibabel.load(self.root / file)
        intensities = np.asarray(img.dataobj)

        # convert 8-bit to 12-bit
        if intensities.min() == 0 and intensities.max() <= 255:
            print('Converted from 12-bit to 8-bit', file.name)
            center = intensities[intensities > 0].mean()
            intensities = (intensities - center) / 255 * 4095

        print(file)

        return {
            'scan_file': file.as_posix(),
            'intensities': intensities.astype(np.int16),
            'spacing': np.array(img.header.get_zooms()),
            'orientation': nibabel.io_orientation(img.affine),
            'shape': np.array(img.header.get_data_shape()),
        }

    def load_targets(
        self,
        *files: Tuple[Path, ...],
    ) -> Dict[str, NDArray[np.bool8]]:
        seg_file = files[0].as_posix()
        seg = nibabel.load(self.root / seg_file)
        labels = np.asarray(seg.dataobj).astype(np.int8)  # keep at int8 for FDI map!!

        assert 'Filtered_Classes' not in seg_file
        assert 'ToothFairy2_Teeth_Dataset' not in seg_file

        jaws = (labels == 1) | (labels == 2)
        if 'ToothFairy2' in seg_file:
            teeth = TeethSegDataset.CLASS_MAP[np.clip(labels - 2, 0, 32)]
        elif 'Dataset164_All_Classes' in seg_file:
            teeth = TeethSegDataset.CLASS_MAP[np.clip(labels - 4, 0, 32)]

        data_dict = {
            'seg_file': seg_file,
            'mandible': jaws,
            'labels': teeth,
        }

        if len(files) == 1:
            return data_dict
        
        representations = []
        for file in files[1:]:
            instances = nibabel.load(self.root / file)
            try:
                instances = np.asarray(instances.dataobj)
                representations.append(instances)
            except Exception as e:
                print('Error:', file)
                raise e

        data_dict['instances'] = np.stack(representations).astype(np.uint8)

        return data_dict
