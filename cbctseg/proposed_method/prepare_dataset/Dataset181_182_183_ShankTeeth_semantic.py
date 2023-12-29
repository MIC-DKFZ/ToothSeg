import shutil
from multiprocessing import Pool, Queue, Process, set_start_method, cpu_count
from time import time
from typing import Tuple

import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import compute_new_shape
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


OVERWRITE_EXISTING = False


def producer(image_fnames, seg_fnames, target_image, target_label, target_queue: Queue):
    for i, s, ti, tl in zip(image_fnames, seg_fnames, target_image, target_label):
        if not OVERWRITE_EXISTING and isfile(ti) and isfile(tl):
            continue
        im_source = sitk.ReadImage(i)
        seg_source = sitk.ReadImage(s)

        source_spacing = im_source.GetSpacing()
        source_origin = im_source.GetOrigin()
        source_direction = im_source.GetDirection()

        im_source = sitk.GetArrayFromImage(im_source).astype(np.float32)

        source_spacing_s = seg_source.GetSpacing()
        source_origin_s = seg_source.GetOrigin()
        source_direction_s = seg_source.GetDirection()
        assert source_spacing == source_spacing_s
        assert source_origin == source_origin_s
        assert source_direction == source_direction_s

        seg_source = sitk.GetArrayFromImage(seg_source).astype(np.uint8)
        target_queue.put((im_source, seg_source, source_spacing, source_origin, source_direction, ti, tl))
    target_queue.put('end')


def resample_core(source_queue: Queue,
                  num_workers: int,
                  export_pool: Pool,
                  target_spacing: Tuple[float, ...] = (0.3, 0.3, 0.3)):
    num_cpu_threads = max((1, cpu_count() - 2, cpu_count() // 2))
    print(num_cpu_threads)
    r = []
    end_ctr = 0
    while True:
        item = source_queue.get()
        if item == 'end':
            end_ctr += 1
            if end_ctr == num_workers:
                break
            continue
        #print('get item')
        im_source, seg_source, source_spacing, source_origin, source_direction, target_image, target_label = item
        source_shape = im_source.shape

        # resample image
        target_shape = compute_new_shape(source_shape, list(source_spacing)[::-1], target_spacing)

        print(f'{os.path.basename(target_label)}; source shape: {source_shape}, target shape {target_shape}')
        try:
            seg_target_correct_labels = \
            resample_torch(torch.from_numpy(seg_source)[None], target_shape, None, None, is_seg=True, num_threads=num_cpu_threads,
                           device=torch.device('cuda:0'))[0].numpy()
        except:
            seg_target_correct_labels = \
            resample_torch(torch.from_numpy(seg_source)[None], target_shape, None, None, is_seg=True, num_threads=num_cpu_threads,
                           device=torch.device('cpu'))[0].numpy()
        torch.cuda.empty_cache()

        seg_target_itk = sitk.GetImageFromArray(seg_target_correct_labels)
        seg_target_itk.SetSpacing(tuple(list(target_spacing)[::-1]))
        seg_target_itk.SetOrigin(source_origin)
        seg_target_itk.SetDirection(source_direction)

        # now resample images. For simplicity, just make this linear
        try:
            im_source = \
                resample_torch(torch.from_numpy(im_source)[None], target_shape, None, None, is_seg=False, num_threads=num_cpu_threads,
                               device=torch.device('cuda:0'))[0].numpy()
        except:
            im_source = \
                resample_torch(torch.from_numpy(im_source)[None], target_shape, None, None, is_seg=False,
                               num_threads=num_cpu_threads,
                               device=torch.device('cpu'))[0].numpy()
        torch.cuda.empty_cache()

        # export image
        im_target = sitk.GetImageFromArray(im_source)
        im_target.SetSpacing(tuple(list(target_spacing)[::-1]))
        im_target.SetOrigin(source_origin)
        im_target.SetDirection(source_direction)

        r1 = export_pool.starmap_async(sitk.WriteImage, ((im_target, target_image),))
        r2 = export_pool.starmap_async(sitk.WriteImage, ((seg_target_itk, target_label),))
        r.append((r1, r2))
    return r


def convert_dataset(source_dir, target_name, target_spacing):
    pool = Pool(4)
    num_processes_loading = 4

    output_dir_base = join(nnUNet_raw, target_name)
    maybe_mkdir_p(join(output_dir_base, 'imagesTr'))
    maybe_mkdir_p(join(output_dir_base, 'imagesTs'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTr'))
    maybe_mkdir_p(join(output_dir_base, 'labelsTs'))

    st = time()
    image_fnames = []
    seg_fnames = []
    target_images = []
    target_labels = []
    source_cases = nifti_files(join(source_dir, 'labelsTr'), join=False)
    for s in source_cases:
        image_fnames.append(join(source_dir, 'imagesTr', s[:-7] + '_0000.nii.gz'))
        seg_fnames.append(join(source_dir, 'labelsTr', s))
        target_images.append(join(nnUNet_raw, target_name, 'imagesTr', s[:-7] + '_0000.nii.gz'))
        target_labels.append(join(nnUNet_raw, target_name, 'labelsTr', s))
    source_test_cases = nifti_files(join(source_dir, 'labelsTs'), join=False)
    for s in source_test_cases:
        image_fnames.append(join(source_dir, 'imagesTs', s[:-7] + '_0000.nii.gz'))
        seg_fnames.append(join(source_dir, 'labelsTs', s))
        target_images.append(join(nnUNet_raw, target_name, 'imagesTs', s[:-7] + '_0000.nii.gz'))
        target_labels.append(join(nnUNet_raw, target_name, 'labelsTs', s))

    processes = []
    q = Queue(maxsize=2)
    for p in range(num_processes_loading):
        pr = Process(target=producer, args=(
            image_fnames[p::num_processes_loading],
            seg_fnames[p::num_processes_loading],
            target_images[p::num_processes_loading],
            target_labels[p::num_processes_loading],
            q), daemon=True)
        pr.start()
        processes.append(
            pr
        )
    r = resample_core(q, num_processes_loading, export_pool=pool, target_spacing=target_spacing)

    _ = [i.get() for j in r for i in j if i is not None]
    print(time() - st)
    shutil.copy(join(source_dir, 'dataset.json'), join(output_dir_base, 'dataset.json'))


if __name__ == '__main__':
    # export nnUNet_raw="/media/isensee/My Book1/datasets/Shank"

    set_start_method('spawn')
    source_dir = join(nnUNet_raw, maybe_convert_to_dataset_name(164))

    convert_dataset(source_dir, f'Dataset{181}_CBCTTeeth_semantic_spacing03', (0.3, 0.3, 0.3))
    convert_dataset(source_dir, f'Dataset{182}_CBCTTeeth_semantic_spacing05', (0.5, 0.5, 0.5))
    convert_dataset(source_dir, f'Dataset{183}_CBCTTeeth_semantic_spacing02', (0.2, 0.2, 0.2))




