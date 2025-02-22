import gc
from multiprocessing import Pool, Queue, Process, set_start_method, cpu_count
from time import time
from typing import Tuple

import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import compute_new_shape
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets, \
    get_identifiers_from_splitted_dataset_folder, create_lists_from_splitted_dataset_folder

OVERWRITE_EXISTING = False


def producer(image_fnames, seg_fnames, target_image, target_label, target_queue: Queue):
    for i, s, ti, tl in zip(image_fnames, seg_fnames, target_image, target_label):
        print(f'{os.path.basename(i)}')
        if not OVERWRITE_EXISTING and isfile(ti) and isfile(tl):
            continue
        print(f'loading {os.path.basename(i)}')
        im_source = sitk.ReadImage(i)
        seg_source = sitk.ReadImage(s)

        source_spacing = im_source.GetSpacing()
        source_origin = im_source.GetOrigin()
        source_direction = im_source.GetDirection()

        im_source = sitk.GetArrayFromImage(im_source).astype(np.float32)

        source_spacing_s = seg_source.GetSpacing()
        # source_origin_s = seg_source.GetOrigin()
        # source_direction_s = seg_source.GetDirection()
        assert source_spacing == source_spacing_s
        # assert source_origin == source_origin_s
        # assert source_direction == source_direction_s

        seg_source = sitk.GetArrayFromImage(seg_source).astype(np.uint8)
        target_queue.put((im_source, seg_source, source_spacing, source_origin, source_direction, ti, tl))
        del im_source, seg_source, source_spacing, source_origin, source_direction, ti, tl
        gc.collect()
    target_queue.put('end')


def resample_core(source_queue: Queue,
                  num_workers: int,
                  export_pool: Pool,
                  target_spacing: Tuple[float, ...] = (0.3, 0.3, 0.3), processes=None):
    with torch.no_grad():
        num_cpu_threads = max((1, cpu_count() - 2, cpu_count() // 2))
        print(f"Num CPU Threads: {num_cpu_threads}")
        r = []
        end_ctr = 0
        while True:
            item = source_queue.get()
            if item == 'end':
                end_ctr += 1
                if end_ctr == num_workers:
                    print('done')
                    break
                continue
            # print('get item')
            im_source, seg_source, source_spacing, source_origin, source_direction, target_image, target_label = item
            source_shape = im_source.shape

            # resample image
            target_shape = compute_new_shape(source_shape, list(source_spacing)[::-1], target_spacing)

            print(f'{os.path.basename(target_label)}; source shape: {source_shape}, target shape {target_shape}')
            seg_target_correct_labels = None
            try:
                seg_target_correct_labels = \
                    resample_torch_simple(torch.from_numpy(seg_source)[None], target_shape, is_seg=True,
                                   num_threads=num_cpu_threads,
                                   device=torch.device('cuda:0'))[0]
                seg_target_correct_labels = seg_target_correct_labels.cpu().numpy()
            except:
                del seg_target_correct_labels
                seg_target_correct_labels = \
                    resample_torch_simple(torch.from_numpy(seg_source)[None], target_shape, is_seg=True,
                                   num_threads=num_cpu_threads,
                                   device=torch.device('cpu'))[0]
                seg_target_correct_labels = seg_target_correct_labels.cpu().numpy()
            torch.cuda.empty_cache()

            seg_target_itk = sitk.GetImageFromArray(seg_target_correct_labels)
            seg_target_itk.SetSpacing(tuple(list(target_spacing)[::-1]))
            seg_target_itk.SetOrigin(source_origin)
            seg_target_itk.SetDirection(source_direction)

            # now resample images. For simplicity, just make this linear
            im_target = None
            try:
                im_target = \
                    resample_torch_simple(torch.from_numpy(im_source)[None], target_shape, is_seg=False,
                                   num_threads=num_cpu_threads,
                                   device=torch.device('cuda:0'))[0]
                im_target = im_target.cpu().numpy()
            except:
                del im_target
                im_target = \
                    resample_torch_simple(torch.from_numpy(im_source)[None], target_shape, is_seg=False,
                                   num_threads=num_cpu_threads,
                                   device=torch.device('cpu'))[0]
                im_target = im_target.cpu().numpy()
            torch.cuda.empty_cache()

            # export image
            im_target = sitk.GetImageFromArray(im_target)
            im_target.SetSpacing(tuple(list(target_spacing)[::-1]))
            im_target.SetOrigin(source_origin)
            im_target.SetDirection(source_direction)

            r1 = export_pool.starmap_async(sitk.WriteImage, ((im_target, target_image),))
            r2 = export_pool.starmap_async(sitk.WriteImage, ((seg_target_itk, target_label),))
            r.append((r1, r2))
            del im_target, target_image, seg_target_itk, target_label, im_source, seg_source
            gc.collect()
    return r


def convert_dataset(source_dir, target_name, target_spacing, num_processes_loading: int = 1, num_processes_export: int = 4):
    pool = Pool(num_processes_export)

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

    dataset = get_filenames_of_train_images_and_targets(source_dir)
    dsj = load_json(join(source_dir, 'dataset.json'))
    fe = dsj['file_ending']

    for k in dataset.keys():
        assert len(dataset[k]['images']) == 1, "this script only supports one input modality for now"
        image_fnames.append(dataset[k]['images'][0])
        seg_fnames.append(dataset[k]['label'])
        # target names are nifti. Other formats (mha) suck
        target_images.append(join(nnUNet_raw, target_name, 'imagesTr', k + '_0000.nii.gz'))
        target_labels.append(join(nnUNet_raw, target_name, 'labelsTr', k + '.nii.gz'))

    imagesTs_dir_source = join(nnUNet_raw, source_dir, 'imagesTs')
    if isdir(imagesTs_dir_source):
        assert isdir(join(nnUNet_raw, source_dir, 'labelsTs')), 'This script expects test set labels to be present if test images are available. Stupid, I know.'
        test_identifiers = get_identifiers_from_splitted_dataset_folder(imagesTs_dir_source, fe)
        lol = create_lists_from_splitted_dataset_folder(imagesTs_dir_source, fe, test_identifiers)

        for li, te in zip(lol, test_identifiers):
            assert len(li) == 1, "this script only supports one input modality for now"

            image_fnames.append(li[0])
            seg_fnames.append(join(source_dir, 'labelsTs', te + fe))
            target_images.append(join(nnUNet_raw, target_name, 'imagesTs', te + '_0000.nii.gz'))
            target_labels.append(join(nnUNet_raw, target_name, 'labelsTs', te + '.nii.gz'))

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
    r = resample_core(q, num_processes_loading, export_pool=pool, target_spacing=target_spacing, processes=processes)

    _ = [i.get() for j in r for i in j if i is not None]
    print(f"Time: {time() - st}")

    if 'dataset' in dsj.keys():
        del dsj['dataset']

    save_json(dsj, join(output_dir_base, 'dataset.json'), sort_keys=False)

    for p in processes:
        p.join()
    pool.close()
    pool.join()
    q.close()


if __name__ == '__main__':
    # export nnUNet_raw="/media/isensee/My Book1/datasets/Shank"

    set_start_method('spawn')
    source_dir = join(nnUNet_raw, maybe_convert_to_dataset_name(164))

    convert_dataset(source_dir, f'Dataset{181}_CBCTTeeth_semantic_spacing03', (0.3, 0.3, 0.3),
                    2, 6)
