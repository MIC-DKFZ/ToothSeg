from multiprocessing import Pool, set_start_method

import numpy as np
import SimpleITK as sitk
import torch.cuda
from batchgenerators.utilities.file_and_folder_operations import *
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg, \
    postprocess_instance_segmentation
import pandas as pd


def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


def convert_all_sem_to_instance(border_core_seg_folder, output_folder, small_center_threshold=0.03,
                                isolated_border_as_separate_instance_threshold=0.03, num_processes: int = 12,
                                overwrite: bool = True, min_instance_size: float = 0,
                                file_ending: str = '.nii.gz'):
    maybe_mkdir_p(output_folder)

    input_files = subfiles(border_core_seg_folder, join=False, suffix=file_ending)
    output_files = [join(output_folder, i) for i in input_files]
    input_files = [join(border_core_seg_folder, i) for i in input_files]

    p = Pool(num_processes)
    res = p.starmap_async(
        load_convert_semantic_to_instance_save,
        zip(input_files,
            output_files,
            [small_center_threshold] * len(output_files),
            [isolated_border_as_separate_instance_threshold] * len(output_files),
            [overwrite] * len(output_files),
            [min_instance_size] * len(output_files),
            )
    )
    _ = res.get()
    p.close()
    p.join()


def load_convert_semantic_to_instance_save(input_file: str, output_file: str, small_center_threshold=0.03,
                                           isolated_border_as_separate_instance_threshold=0.03, overwrite=True,
                                           min_instance_size: float = 0):
    if overwrite or not isfile(output_file):
        print(os.path.basename(input_file))
        itk_img = sitk.ReadImage(input_file)
        npy_img = sitk.GetArrayFromImage(itk_img)
        spacing = np.array(itk_img.GetSpacing())[::-1]
        instance_seg = convert_semantic_to_instanceseg(npy_img, spacing, small_center_threshold,
                                                       isolated_border_as_separate_instance_threshold)
        if min_instance_size > 0:
            vol_per_voxel = np.prod(spacing)
            n_pixel_cutoff = min_instance_size / vol_per_voxel
            instances = [i for i in pd.unique(instance_seg.ravel()) if i != 0]
            for i in instances:
                mask = instance_seg == i
                if np.sum(mask) < n_pixel_cutoff:
                    instance_seg[mask] = 0
        instance_seg = postprocess_instance_segmentation(instance_seg)
        itk_res = sitk.GetImageFromArray(instance_seg)
        itk_res = copy_geometry(itk_res, itk_img)
        sitk.WriteImage(itk_res, output_file)


if __name__ == '__main__':
    set_start_method('spawn')

    small_center_threshold_default = 20  # =0.3**3 * 10  # equivalent to 10 pixels at 0.3 spacing
    isolated_border_as_separate_instance_threshold_default = 0  # =0.3**3 * 10  # equivalent to 10 pixels at 0.3 spacing
    min_instance_size_default = 20

    import argparse
    parser = argparse.ArgumentParser("This script takes a folder containing nifti files with border-core predictions "
                                     "and converts them into an instance segmentation map.")
    parser.add_argument('-i', type=str, required=True,
                        help='Input folder. Must contain nifti files with border-core segmentations')
    parser.add_argument('-o', type=str, required=True,
                        help="Output folder. Must be empty! If it doesn't exist it will be created")
    parser.add_argument('-np', type=int, required=False, default=8,
                        help='Number of processes used for multiprocessing. Default: 8. Make this a lot higher if you '
                             'can!')
    parser.add_argument('-sct', type=float, required=False, default=small_center_threshold_default,
                        help=f'Small center threshold (volume). Removes small center predictions. Default: '
                             f'{small_center_threshold_default}')
    parser.add_argument('-ibsi', type=float, required=False, default=isolated_border_as_separate_instance_threshold_default,
                        help=f'Isolated border predictions (no core) larger than this (volume) will be made a separate '
                             f'instance instead of being deleted. Default: '
                             f'{isolated_border_as_separate_instance_threshold_default}')
    parser.add_argument('-fe', type=str, required=False, default='.nii.gz',
                        help=f'File ending, Default: .nii.gz')
    parser.add_argument('-min_inst_size', type=float, required=False, default=min_instance_size_default,
                        help=f'Minimum instance size (volume). Default: '
                             f'{min_instance_size_default}')
    parser.add_argument('--overwrite_existing', action='store_true',
                        help='By default the script will skip existing results. Set this flag to overwrite (recompute) '
                             'them instead.')
    args = parser.parse_args()

    convert_all_sem_to_instance(args.i, args.o, overwrite=args.overwrite_existing,
                                small_center_threshold=args.sct,
                                isolated_border_as_separate_instance_threshold=
                                args.ibsi,
                                num_processes=args.np, min_instance_size=args.min_inst_size,
                                file_ending=args.fe)

    #
    # folders_pred = [
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    # ]
    #
    # num_processes = 128
    # for f in folders_pred:
    #     convert_all_sem_to_instance(f, f + '_instances', overwrite=False,
    #                                 small_center_threshold=small_center_threshold_default,
    #                                 isolated_border_as_separate_instance_threshold=
    #                                 isolated_border_as_separate_instance_threshold_default,
    #                                 num_processes=num_processes)

