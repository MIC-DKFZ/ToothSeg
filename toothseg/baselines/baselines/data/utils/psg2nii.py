from collections import defaultdict
from functools import partial
import multiprocessing as mp
from pathlib import Path

import pycocotools.mask as maskUtils
import nibabel
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from tqdm import tqdm


class PSG:

    COUNT = 0

    def __init__(
        self,
        version: int,
        image_id: str,
        shape: tuple[int, int, int],
        obj_type: str,
        obj_id: str,
        content: NDArray[np.bool_],
    ):
        self.version = version
        self.image_id = image_id
        self.shape = shape
        self.type = obj_type
        self.id = obj_id
        self.content = content

        assert version == 2

    def to_numpy(self) -> NDArray[np.bool_]:
        out = self.content.reshape(self.shape)
        out = out.transpose(2, 1, 0)[::-1]

        return out
    
    def to_coco(self) -> dict:
        rle = maskUtils.encode(self.to_numpy()[0])
        PSG.COUNT += 1

        return {
            'id': PSG.COUNT,
            'image_id': self.image_id,
            'category_id': f'{self.type}_{self.id}',
            'bbox': maskUtils.toBbox(rle).tolist(),
            'area': maskUtils.area(rle).item(),
            'segmentation': {
                'size': rle['size'],
                'counts': rle['counts'].decode(),
            },
            'iscrowd': 0,
        }

    @staticmethod
    def read_file(path: Path):
        image_id = path.parent.parent.parent.name
        fp = open(path, 'rb')

        version = int.from_bytes(fp.read(1), byteorder='big')

        shape = ()
        for _ in range(3):
            dim = int.from_bytes(fp.read(2), byteorder='big')
            shape = shape + (dim,)

        object_props = []
        for _ in range(2):
            str_size = int.from_bytes(fp.read(1), byteorder='big')
            str_encoded = fp.read(str_size)
            string = str_encoded.decode('utf-8')
            object_props.append(string)

        object_type, object_id = object_props

        out = np.zeros(np.prod(shape), dtype=bool)
        while True:
            chunk_start = fp.read(4)
            if chunk_start:
                start = int.from_bytes(chunk_start, byteorder='big')
            else:
                break

            chunk_end = fp.read(4)
            if chunk_end:
                end = int.from_bytes(chunk_end, byteorder='big')
                out[start:end] = True
            else:
                out[start:] = True
                break
        
        fp.close()

        return PSG(version, image_id, shape, object_type, object_id, out)
    

def process_case(root1, root2, classes, scan_file):

    
    if scan_file.name != 'suzanna-mobile-kite_0000.nii.gz':
        return scan_file

    psg_dir = root1 / scan_file.name.split('_')[0] / 'psg_manual_ann'
    out_file = root2 / 'labelsTr2' / f'{psg_dir.parent.name}.nii.gz'

    # if out_file.exists():
    #     return scan_file

    old_seg_file = root2 / 'labelsTr' / f'{psg_dir.parent.name}.nii.gz'
    old_seg_nii = nibabel.load(old_seg_file)
    old_seg = np.asarray(old_seg_nii.dataobj)

    seg = None
    for psg_file in sorted(psg_dir.glob('**/*.psg')):
        if seg is not None and classes[psg_file.stem] == 0:
            continue

        psg = PSG.read_file(psg_file)
        mask = psg.to_numpy()
        if mask.sum() < 1000:
            continue

        if seg is None:
            seg = np.zeros_like(mask).astype(int)

        seg[mask] = classes[psg_file.stem]

    if seg is None:
        k = 3

    saved = False
    for transpose in [True, False]:
        if transpose:
            new_seg = np.transpose(seg, (2, 1, 0))
        else:
            new_seg = seg

        if np.any(np.array(new_seg.shape) != old_seg.shape):
            continue

        if np.all(old_seg[new_seg > 0] > 0):
            seg_nii = nibabel.Nifti1Image(new_seg.astype(np.uint8), old_seg_nii.affine)
            nibabel.save(seg_nii, root2 / 'labelsTs2' / f'{psg_dir.parent.name}.nii.gz')
            saved = True
            break
    
    if not saved:
        print(scan_file)

    return scan_file



if __name__ == '__main__':
    classes = defaultdict(int, {
        **{
            f'TOOTH_{q}{e}': 10 * q + e
            for q in range(1, 5)
            for e in range(1, 9)
        },
        **{
            f'PRIMARY_TOOTH_{q}{e}': 10 * q + e
            for q in range(5, 9)
            for e in range(1, 6)
        },
        **{
            f'DENTAL_IMPLANT_{q}{e}': 10 * q + e
            for q in range(1, 5)
            for e in range(1, 9)
        },
        **{
            f'NON_TOOTH_SUPPORTED_CROWN_{q}{e}': 10 * q + e
            for q in range(1, 5)
            for e in range(1, 9)
        },
    })

    classes = defaultdict(int, {
        'LOWER_JAW': 1,
        'UPPER_JAW': 2,
        **{
            f'DENTAL_IMPLANT_{q}{e}': 3
            for q in range(1, 5)
            for e in range(1, 9)
        },
        **{
            f'NON_TOOTH_SUPPORTED_CROWN_{q}{e}': 4
            for q in range(1, 5)
            for e in range(1, 9)
        },
        **{
            f'TOOTH_{q}{e}': 5 + i * 8 + j
            for i, q in enumerate(range(1, 5))
            for j, e in enumerate(range(1, 9))
        },
        **{
            f'PRIMARY_TOOTH_{q}{e}': 37 + i * 5 + j
            for i, q in enumerate(range(5, 9))
            for j, e in enumerate(range(1, 6))
        },
    })

    root1 = Path('/mnt/diag/Fabian')
    root2 = Path('/home/mkaailab/Documents/CBCT/baselines/data/train')
    with mp.Pool(24) as p:
        func = partial(process_case, root1, root2, classes)
        scan_files = sorted(root2.glob('**/*_0000.nii.gz'))
        t = tqdm(p.imap_unordered(func, scan_files), total=len(scan_files))
        for scan_file in t:
            t.set_description(scan_file.name)
