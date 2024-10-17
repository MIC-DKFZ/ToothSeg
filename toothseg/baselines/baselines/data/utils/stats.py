from pathlib import Path

import nibabel
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/tooth_segmentation/data/Dataset164_Filtered_Classes/labelsTr')

    all_lengths = []
    for nii_path in root.glob('*.nii.gz'):
        nii = nibabel.load(nii_path)
        labels = np.asarray(nii.dataobj)
        spacing = np.linalg.norm(nii.affine[:, :3], axis=0)

        orientation = nibabel.io_orientation(nii.affine)
        orientation = orientation.astype(int)

        lengths = np.column_stack(np.nonzero(labels)).ptp(0) * spacing
        lengths = lengths[orientation[:, 0]]
        print(lengths)

        all_lengths.append(lengths)

    lengths = np.stack(all_lengths)
    print(lengths.max(0))




