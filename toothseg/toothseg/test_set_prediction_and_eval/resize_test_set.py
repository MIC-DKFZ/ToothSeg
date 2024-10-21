from multiprocessing import Pool, Queue, Process, cpu_count, set_start_method
from typing import Tuple

import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import compute_new_shape
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

OVERWRITE_EXISTING = False


def _resample_core_resize_folder(source_queue: Queue,
                                 num_workers: int,
                                 export_pool: Pool,
                                 target_spacing: Tuple[float, ...] = (0.3, 0.3, 0.3),
                                 num_cpu_threads: int = max((1, cpu_count() - 2, cpu_count() // 2))):
    r = []
    end_ctr = 0
    while True:
        item = source_queue.get()
        if item == 'end':
            end_ctr += 1
            if end_ctr == num_workers:
                break
            continue
        # print('get item')
        im_source, source_spacing, source_origin, source_direction, infile, ofile = item
        source_shape = im_source.shape

        # resample image
        target_shape = compute_new_shape(source_shape, list(source_spacing)[::-1], target_spacing)

        print(f'{os.path.basename(ofile)}; source shape: {source_shape}, target shape {target_shape}, '
              f'source spacing: {list(source_spacing)[::-1]}, target spacing {target_spacing}')

        # now resample images. For simplicity, just make this linear
        try:
            im_source = \
                resample_torch_simple(torch.from_numpy(im_source)[None], target_shape, is_seg=False,
                               num_threads=num_cpu_threads,
                               device=torch.device('cuda:0'))[0].numpy()
        except:
            im_source = \
                resample_torch_simple(torch.from_numpy(im_source)[None], target_shape, is_seg=False,
                               num_threads=num_cpu_threads,
                               device=torch.device('cpu'))[0].numpy()
        torch.cuda.empty_cache()

        # export image
        im_target = sitk.GetImageFromArray(im_source)
        im_target.SetSpacing(tuple(list(target_spacing)[::-1]))
        im_target.SetOrigin(source_origin)
        im_target.SetDirection(source_direction)

        r1 = export_pool.starmap_async(sitk.WriteImage, ((im_target, ofile),))
        r.append(r1)
    return r


def _producer_resize_folder(input_files, output_files, target_queue: Queue):
    for infile, ofile in zip(input_files, output_files):
        if not OVERWRITE_EXISTING and isfile(ofile):
            continue
        im_source = sitk.ReadImage(infile)

        source_spacing = im_source.GetSpacing()
        source_origin = im_source.GetOrigin()
        source_direction = im_source.GetDirection()

        im_source = sitk.GetArrayFromImage(im_source).astype(np.float32)
        target_queue.put((im_source, source_spacing, source_origin, source_direction, infile, ofile))
    target_queue.put('end')


def resize_folder(input_folder,
                  output_folder,
                  target_spacing: Tuple[float, float, float],
                  num_processes_loading: int = 4,
                  num_processes_export: int = 4,
                  resize_num_cpu_threads: int = max((1, cpu_count() - 2, cpu_count() // 2))):
    """
    THIS ONLY WORKS FOR IMAGES. DO NOT (!!!!) USE THIS FOR RESIZING SEGMENTATIONS
    :param input_folder:
    :param output_folder:
    :param target_spacing:
    :param num_processes_loading:
    :param num_processes_export:
    :return:
    """
    pool = Pool(num_processes_export)
    maybe_mkdir_p(output_folder)

    source_cases = nifti_files(input_folder, join=False)
    output_cases = [join(output_folder, s) for s in source_cases]
    source_cases = [join(input_folder, s) for s in source_cases]

    processes = []
    q = Queue(maxsize=2)
    for p in range(num_processes_loading):
        pr = Process(target=_producer_resize_folder,
                     args=(
                         source_cases[p::num_processes_loading],
                         output_cases[p::num_processes_loading],
                         q),
                     daemon=True)
        pr.start()
        processes.append(
            pr
        )
    r = _resample_core_resize_folder(
        q,
        num_processes_loading,
        export_pool=pool,
        target_spacing=target_spacing,
        num_cpu_threads=resize_num_cpu_threads
    )

    _ = [i.get() for i in r if i is not None]


if __name__ == '__main__':
    set_start_method('spawn')

    import argparse
    parser = argparse.ArgumentParser(
        "This script takes a folder containing nifti files and resizes them to corresponding "
        "(equally named) niftis on the ref folder. The target spacing is 0.2x0.2x0.2")
    parser.add_argument('-i', type=str, required=True,
                        help='Input folder. Must contain nifti files')
    parser.add_argument('-o', type=str, required=True,
                        help="Output folder. Must be empty! If it doesn't exist it will be created")
    args = parser.parse_args()
    # source_folder = join(nnUNet_raw, maybe_convert_to_dataset_name(164), 'imagesTs')
    # target_folder = join(nnUNet_raw, maybe_convert_to_dataset_name(164),
    #                      'imagesTs_resized_for_instanceseg_spacing_02_02_02')
    source_folder = args.i
    target_folder = args.o
    resize_folder(source_folder, target_folder, target_spacing=(0.2, 0.2, 0.2),
                  num_processes_loading=8,
                  num_processes_export=32)

