import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import pandas as pd
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_instanceseg_to_semantic_patched
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from scipy.ndimage import binary_fill_holes


def semseg_to_instanceseg(source_file: str, target_file: str, border_thickness_in_mm):
    seg_itk = sitk.ReadImage(source_file)
    current_spacing = list(seg_itk.GetSpacing())[::-1]
    seg = sitk.GetArrayFromImage(seg_itk)
    seg[seg <= 4] = 0  # remove jaws and prosthetics
    instances = np.sort(pd.unique(seg.ravel()))
    # small holes in the reference segmentation are super annoying because the spawn a large ring of order around
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


if __name__ == '__main__':
    # we start with the raw Dataset162. We convert all images and segmentations to spacing 0.3.
    # this is necessary so that we do not destroy the border-core segmentation when resampling back to the original
    # image shape. Conversion back to instance must be performed at the target spacing
    source_dataset_id = 181
    source_dataset_name = maybe_convert_to_dataset_name(source_dataset_id)

    target_dataset_id = 183
    target_dataset_name = f'Dataset{target_dataset_id}_ShankTeethv2_instance_spacing03'

    border_thickness_in_mm = 2 * 0.3

    p = Pool(16)

    output_dir_base = join(nnUNet_raw, target_dataset_name)
    maybe_mkdir_p(join(output_dir_base, 'imagesTr'))
    maybe_mkdir_p(join(output_dir_base, 'imagesTs'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTr'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTs'))

    source_cases = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)
    r = []
    for s in source_cases:
        r.append(
            p.starmap_async(
                semseg_to_instanceseg,
                ((
                     join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                     join(nnUNet_raw, target_dataset_name, 'labelsTr', s),
                     border_thickness_in_mm
                 ),)
            )
        )
        shutil.copy(
            join(nnUNet_raw, source_dataset_name, 'imagesTr', s[:-7] + '_0000.nii.gz'),
            join(nnUNet_raw, target_dataset_name, 'imagesTr', s[:-7] + '_0000.nii.gz'),

        )
    source_cases = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTs'), join=False)
    for s in source_cases:
        r.append(
            p.starmap_async(
                semseg_to_instanceseg,
                ((
                     join(nnUNet_raw, source_dataset_name, 'labelsTs', s),
                     join(nnUNet_raw, target_dataset_name, 'labelsTs', s),
                     border_thickness_in_mm

                 ),)
            )
        )
        shutil.copy(
            join(nnUNet_raw, source_dataset_name, 'imagesTs', s[:-7] + '_0000.nii.gz'),
            join(nnUNet_raw, target_dataset_name, 'imagesTs', s[:-7] + '_0000.nii.gz'),

        )
    _ = [i.get() for i in r]
    p.close()
    p.join()
    generate_dataset_json(join(nnUNet_raw, target_dataset_name), {0: 'CT'}, {f'{i}': i for i in range(3)},
                          910, '.nii.gz', target_dataset_name)

    #########################################################################################
    # 184 gets a thiccer border
    target_dataset_id = 184
    target_dataset_name = f'Dataset{target_dataset_id}_ShankTeethv2_instance_spacing03_brd3'

    border_thickness_in_mm = 3 * 0.3

    p = Pool(8)

    output_dir_base = join(nnUNet_raw, target_dataset_name)
    maybe_mkdir_p(join(output_dir_base, 'imagesTr'))
    maybe_mkdir_p(join(output_dir_base, 'imagesTs'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTr'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTs'))

    source_cases = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)
    r = []
    for s in source_cases:
        r.append(
            p.starmap_async(
                semseg_to_instanceseg,
                ((
                     join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                     join(nnUNet_raw, target_dataset_name, 'labelsTr', s),
                     border_thickness_in_mm
                 ),)
            )
        )
        shutil.copy(
            join(nnUNet_raw, source_dataset_name, 'imagesTr', s[:-7] + '_0000.nii.gz'),
            join(nnUNet_raw, target_dataset_name, 'imagesTr', s[:-7] + '_0000.nii.gz'),

        )
    source_cases = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTs'), join=False)
    for s in source_cases:
        r.append(
            p.starmap_async(
                semseg_to_instanceseg,
                ((
                     join(nnUNet_raw, source_dataset_name, 'labelsTs', s),
                     join(nnUNet_raw, target_dataset_name, 'labelsTs', s),
                     border_thickness_in_mm

                 ),)
            )
        )
        shutil.copy(
            join(nnUNet_raw, source_dataset_name, 'imagesTs', s[:-7] + '_0000.nii.gz'),
            join(nnUNet_raw, target_dataset_name, 'imagesTs', s[:-7] + '_0000.nii.gz'),

        )
    _ = [i.get() for i in r]
    p.close()
    p.join()
    generate_dataset_json(join(nnUNet_raw, target_dataset_name), {0: 'CT'}, {f'{i}': i for i in range(3)},
                          910, '.nii.gz', target_dataset_name)