from pathlib import Path

import nibabel
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/tooth_segmentation/data/cuinetPr')

    out_dir = root.parent / 'cuinetPr_iso'
    out_dir.mkdir(exist_ok=True)

    iso_fdi_map = np.array([
        0,
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
    ])
    fdi_iso_map = np.full(iso_fdi_map.max() + 1, -1)
    fdi_iso_map[iso_fdi_map] = np.arange(iso_fdi_map.shape[0])

    for seg_path in tqdm(list(root.glob('*.nii.gz'))):
        if '0000' not in seg_path.name:
            continue

        out_name = seg_path.name.split('.')[0][:-5] + '.nii.gz'

        seg_nii = nibabel.load(seg_path)
        seg = np.asarray(seg_nii.dataobj)

        seg = fdi_iso_map[seg].astype(np.uint16)

        seg_nii = nibabel.Nifti1Image(seg, seg_nii.affine)

        nibabel.save(seg_nii, seg_path.parent / out_name)



