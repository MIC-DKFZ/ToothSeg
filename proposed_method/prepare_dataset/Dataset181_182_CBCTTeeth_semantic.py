from multiprocessing import Pool
from typing import Tuple

import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from acvl_utils.array_manipulation.resampling import maybe_resample_on_gpu
from nnunetv2.paths import nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from torch.nn import functional as F
import pandas as pd
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

blacklist = [
    'bride-bad-felidae_0000.nii.gz',  # has wrong spacing information. Booms up the RAM
]


def resample_save(source_image: str, source_label: str, target_image: str, target_label: str,
                  target_spacing: Tuple[float, ...] = (0.3, 0.3, 0.3), skip_existing: bool = True,
                  export_pool: Pool = None):
    print(f'{os.path.basename(source_image)}')
    if skip_existing and isfile(target_label) and isfile(target_image):
        return None, None

    if os.path.basename(source_image) in blacklist:
        print(f'skipping {os.path.basename(source_image)} because its in my blacklist. Naughty.')
        return None, None

    seg_source = sitk.GetArrayFromImage(sitk.ReadImage(source_label)).astype(np.uint8)
    if np.any(seg_source > 36):
        print(f'skipping {os.path.basename(source_image)} due to juvenile teeth')
        return None, None

    im_source = sitk.ReadImage(source_image)

    source_spacing = im_source.GetSpacing()
    source_origin = im_source.GetOrigin()
    source_direction = im_source.GetDirection()

    im_source = sitk.GetArrayFromImage(im_source).astype(np.float32)
    source_shape = im_source.shape

    # resample image
    target_shape = compute_new_shape(source_shape, list(source_spacing)[::-1], target_spacing)

    print(f'source shape: {source_shape}, target shape {target_shape}')

    # one hot generation is slow af. Let's do it this way:
    seg_source = torch.from_numpy(seg_source)
    unique_labels = None
    try:
        torch.cuda.empty_cache()
        device = 'cuda:0'
        # having the target array on device will blow up, so we need to have this on CPU
        with torch.no_grad():
            seg_source_gpu = seg_source.to(device)
            unique_labels = torch.unique(seg_source_gpu)
            seg_onehot_target_shape = torch.zeros((len(unique_labels), *target_shape), dtype=torch.float16, device='cpu')
            for i, l in enumerate(unique_labels):
                seg_onehot_target_shape[i] = F.interpolate((seg_source_gpu == l).half()[None, None], tuple(target_shape), mode='trilinear')[0, 0].cpu()
        del seg_source_gpu
    except RuntimeError:
        print('GPU wasnt happy with this resampling. Lets give the CPU a chance to sort it out')
        print(f'source shape {source_shape}, target shape {target_shape}, unique_labels {unique_labels}')
        del seg_source_gpu
        device = 'cpu'
        with torch.no_grad():
            if unique_labels is None:
                unique_labels = torch.unique(seg_source)
            seg_onehot_target_shape = torch.zeros((len(unique_labels), *target_shape), dtype=torch.float16,
                                                  device='cpu')
            for i, l in enumerate(unique_labels):
                # float because half is not implemented on cpu
                seg_onehot_target_shape[i] = F.interpolate(torch.from_numpy(seg_source == l).to(device).float()[None, None], tuple(target_shape), mode='trilinear')[0, 0].cpu()
    finally:
        torch.cuda.empty_cache()

    # ok now argmax
    try:
        device = 'cuda:0'
        seg_onehot_target_shape = seg_onehot_target_shape.to(device)
        seg_onehot_target_shape = torch.argmax(seg_onehot_target_shape, dim=0)
    except RuntimeError:
        print('GPU wasnt happy with this argmax. Now the CPU can show us what she got')
        print(f'seg_onehot_target_shape shape {seg_onehot_target_shape.shape}')
        device = 'cpu'
        seg_onehot_target_shape = seg_onehot_target_shape.to(device)
        seg_onehot_target_shape = torch.argmax(seg_onehot_target_shape, dim=0)

    seg_onehot_target_shape = seg_onehot_target_shape.cpu().numpy()

    seg_target_correct_labels = np.zeros_like(seg_onehot_target_shape, dtype=np.uint8)
    for i, l in enumerate(unique_labels.cpu().numpy()):
        seg_target_correct_labels[seg_onehot_target_shape == i] = l
    del seg_onehot_target_shape

    seg_target_itk = sitk.GetImageFromArray(seg_target_correct_labels)
    seg_target_itk.SetSpacing(tuple(list(target_spacing)[::-1]))
    seg_target_itk.SetOrigin(source_origin)
    seg_target_itk.SetDirection(source_direction)

    # now resample images. For simplicity, just make this linear
    im_source = maybe_resample_on_gpu(torch.from_numpy(im_source[None]), tuple(target_shape), return_type=torch.float,
                                      compute_precision=torch.float, fallback_compute_precision=float)[0].cpu().numpy()

    # export image
    im_target = sitk.GetImageFromArray(im_source)
    im_target.SetSpacing(tuple(list(target_spacing)[::-1]))
    im_target.SetOrigin(source_origin)
    im_target.SetDirection(source_direction)

    if export_pool is None:
        sitk.WriteImage(im_target, target_image)
        sitk.WriteImage(seg_target_itk, target_label)
        return None, None
    else:
        r1 = export_pool.starmap_async(sitk.WriteImage, ((im_target, target_image),))
        r2 = export_pool.starmap_async(sitk.WriteImage, ((seg_target_itk, target_label),))
        return r1, r2


if __name__ == '__main__':
    # we start with the raw Dataset163. We convert all images and segmentations to spacing 0.3/0.5.

    # remember to export OMP_NUM_THREADS=30
    source_dataset_id = 163
    source_dataset_name = maybe_convert_to_dataset_name(source_dataset_id)

    target_dataset_id = 181
    target_dataset_name = f'Dataset{target_dataset_id}_ShankTeethv2_spacing03'
    target_spacing = (0.3, 0.3, 0.3)

    p = Pool(8)

    output_dir_base = join(nnUNet_raw, target_dataset_name)
    maybe_mkdir_p(join(output_dir_base, 'imagesTr'))
    maybe_mkdir_p(join(output_dir_base, 'imagesTs'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTr'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTs'))

    source_cases = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)
    r = []
    for s in source_cases:
        r.append(resample_save(join(nnUNet_raw, source_dataset_name, 'imagesTr', s[:-7] + '_0000.nii.gz'),
                      join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                      join(nnUNet_raw, target_dataset_name, 'imagesTr', s[:-7] + '_0000.nii.gz'),
                      join(nnUNet_raw, target_dataset_name, 'labelsTr', s),
                      target_spacing, skip_existing=True, export_pool=p
                      ))
    maybe_mkdir_p(join(nnUNet_raw, source_dataset_name, 'labelsTs'))
    maybe_mkdir_p(join(nnUNet_raw, source_dataset_name, 'imagesTs'))
    source_cases = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTs'), join=False)
    for s in source_cases:
        r.append(resample_save(join(nnUNet_raw, source_dataset_name, 'imagesTs', s[:-7] + '_0000.nii.gz'),
                      join(nnUNet_raw, source_dataset_name, 'labelsTs', s),
                      join(nnUNet_raw, target_dataset_name, 'imagesTs', s[:-7] + '_0000.nii.gz'),
                      join(nnUNet_raw, target_dataset_name, 'labelsTs', s),
                      target_spacing, skip_existing=True, export_pool=p
                      ))
    _ = [i.get() for j in r for i in j if i is not None]  # oof.
    p.close()
    p.join()
    generate_dataset_json(join(nnUNet_raw, target_dataset_name), {0: 'CT'}, {f'{i}': i for i in range(37)},
                          910, '.nii.gz', target_dataset_name)

    source_dataset_name = maybe_convert_to_dataset_name(source_dataset_id)

    target_dataset_id = 182
    target_dataset_name = f'Dataset{target_dataset_id}_ShankTeethv2_spacing05'
    target_spacing = (0.5, 0.5, 0.5)

    p = Pool(8)

    output_dir_base = join(nnUNet_raw, target_dataset_name)
    maybe_mkdir_p(join(output_dir_base, 'imagesTr'))
    maybe_mkdir_p(join(output_dir_base, 'imagesTs'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTr'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTs'))

    source_cases = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)
    r = []
    for s in source_cases:
        r.append(resample_save(join(nnUNet_raw, source_dataset_name, 'imagesTr', s[:-7] + '_0000.nii.gz'),
                      join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                      join(nnUNet_raw, target_dataset_name, 'imagesTr', s[:-7] + '_0000.nii.gz'),
                      join(nnUNet_raw, target_dataset_name, 'labelsTr', s),
                      target_spacing, skip_existing=True, export_pool=p
                      ))
    source_cases = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTs'), join=False)
    for s in source_cases:
        r.append(resample_save(join(nnUNet_raw, source_dataset_name, 'imagesTs', s[:-7] + '_0000.nii.gz'),
                      join(nnUNet_raw, source_dataset_name, 'labelsTs', s),
                      join(nnUNet_raw, target_dataset_name, 'imagesTs', s[:-7] + '_0000.nii.gz'),
                      join(nnUNet_raw, target_dataset_name, 'labelsTs', s),
                      target_spacing, skip_existing=True, export_pool=p
                      ))
    _ = [i.get() for j in r for i in j if i is not None]
    p.close()
    p.join()
    generate_dataset_json(join(nnUNet_raw, target_dataset_name), {0: 'CT'}, {f'{i}': i for i in range(37)},
                          910, '.nii.gz', target_dataset_name)

