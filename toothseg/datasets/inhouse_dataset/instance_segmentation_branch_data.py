from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import pandas as pd
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_instanceseg_to_semantic_patched
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from scipy.ndimage import binary_fill_holes

from toothseg.datasets.inhouse_dataset.semantic_segmentation_branch_data import convert_dataset

OVERWRITE_EXISTING = False


def semseg_to_instanceseg(source_file: str, target_file: str, border_thickness_in_mm):
    if not OVERWRITE_EXISTING and isfile(target_file):
        return
    seg_itk = sitk.ReadImage(source_file)
    current_spacing = list(seg_itk.GetSpacing())[::-1]
    seg = sitk.GetArrayFromImage(seg_itk)
    instances = np.sort(pd.unique(seg.ravel()))
    # small holes in the reference segmentation are super annoying because the spawn a large ring of border around
    # them. These holes are just annotation errors and these rings will confuse the model. Best to fill those holes.
    for i in instances:
        if i != 0:
            mask = seg == i
            mask_closed = binary_fill_holes(mask)
            seg[mask_closed] = i
    semseg = convert_instanceseg_to_semantic_patched(seg.astype(np.uint8), current_spacing,
                                                     border_thickness_in_mm).astype(np.uint8)
    out_itk = sitk.GetImageFromArray(semseg)
    out_itk.CopyInformation(seg_itk)
    sitk.WriteImage(out_itk, target_file)


def convert_sem_dataset_to_instance(
        source_dataset_name,
        target_dataset_name,
        dataset_spacing,
        border_thickness_in_pixels,
        num_processes: int = 16):
    border_thickness_in_mm = border_thickness_in_pixels * dataset_spacing

    p = Pool(num_processes)
    output_dir_base = join(nnUNet_raw, target_dataset_name)
    maybe_mkdir_p(join(output_dir_base, 'imagesTr'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTr'))

    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, source_dataset_name))
    dsj = load_json(join(nnUNet_raw, source_dataset_name, 'dataset.json'))
    fe = dsj['file_ending']
    r = []
    for k in list(dataset.keys()):
        r.append(
            p.starmap_async(
                semseg_to_instanceseg,
                ((
                     dataset[k]['label'],
                     join(nnUNet_raw, target_dataset_name, 'labelsTr', os.path.basename(dataset[k]['label'])),
                     border_thickness_in_mm
                 ),)
            )
        )

    _ = [i.get() for i in r]
    p.close()
    p.join()

    for k in dataset:
        dataset[k]['images'] = [os.path.join(os.pardir, source_dataset_name,
                                             os.path.relpath(i, join(nnUNet_raw, source_dataset_name)))
                                for i in dataset[k]['images']]
        dataset[k]['label'] = os.path.relpath(dataset[k]['label'], join(nnUNet_raw, source_dataset_name))
    generate_dataset_json(join(nnUNet_raw, target_dataset_name), dsj["channel_names"], {'background': 0, 'center': 1, 'border': 2},
                          len(dataset), fe, target_dataset_name, dataset=dataset)


if __name__ == '__main__':
    source_dataset = maybe_convert_to_dataset_name(164)
    source_dir = join(nnUNet_raw, source_dataset)

    # needed for instance segmentation
    convert_dataset(
        source_dir,
        f'Dataset{165}_CBCTTeeth_semantic_spacing02',
        (0.2, 0.2, 0.2),
        2, 6,
    )

    convert_sem_dataset_to_instance(
        maybe_convert_to_dataset_name(165),
        'Dataset166_CBCTTeeth_instance_spacing02_brd3px',
        0.2,
        3,
        num_processes=32
    )
