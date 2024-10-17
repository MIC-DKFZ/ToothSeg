from multiprocessing import set_start_method
from multiprocessing.pool import ThreadPool
from time import sleep

import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple


def resample_segmentations_to_ref(ref_folder, pred_folder, output_folder, overwrite=False,
                                  num_threads_cpu_resampling=128, threads_loading: int = 16,
                                  threads_saving: int = 8):
    """
    This function circumvents I/O problems through threading. Its fast AF boiiiii

    If _loader is too fast this function can run into OOM problems because it needs to store so many intermediate
    results. If we run into that problem we should just rewrite it to use a queue with fixed length for communication.
    Couldn't be assed to do this right now.
    Cheapo solution is to reduce threads_loading.
    """

    def _loader(casename):
        print(os.path.basename(casename))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(join(pred_folder, casename))).astype(np.uint8)
        target_file = join(ref_folder, casename)
        if not isfile(target_file):
            # when the target file is an image it will have the _0000 suffix
            target_file = target_file[:-7] + '_0000.nii.gz'
        target_itk_img = sitk.ReadImage(target_file)
        target_shape = sitk.GetArrayFromImage(target_itk_img).shape
        target_spacing = target_itk_img.GetSpacing()
        target_direction = target_itk_img.GetDirection()
        target_origin = target_itk_img.GetOrigin()
        return casename, seg, target_shape, target_spacing, target_direction, target_origin

    def _saver(cn, seg_res, target_spacing, target_direction, target_origin):
        seg_resampled_itk = sitk.GetImageFromArray(seg_res)
        seg_resampled_itk.SetDirection(target_direction)
        seg_resampled_itk.SetSpacing(target_spacing)
        seg_resampled_itk.SetOrigin(target_origin)
        sitk.WriteImage(seg_resampled_itk, join(output_folder, cn))

    maybe_mkdir_p(output_folder)
    files = nifti_files(pred_folder, join=False)

    tp = ThreadPool(processes=threads_loading)
    load_res = []
    for f in files:
        if not overwrite and isfile(join(output_folder, f)):
            continue
        load_res.append(
            tp.starmap_async(_loader,
                             ((f, ),))
        )

    tps = ThreadPool(processes=threads_saving)
    save_res = []

    while len(load_res) > 0:
        for i in range(len(load_res)):
            if load_res[i].ready():
                c, s, trg_shp, ts, td, to = load_res.pop(i).get()[0]
                try:
                    print(f'GPU resampling for {c}')
                    seg_resampled = \
                        resample_torch_simple(torch.from_numpy(s)[None], trg_shp, is_seg=True,
                                              num_threads=1,
                                              device=torch.device('cuda:0'))[0].numpy()
                except:
                    print(f'GPU failed. Using CPU for: {c}')
                    seg_resampled = \
                        resample_torch_simple(torch.from_numpy(s)[None], trg_shp, is_seg=True,
                                              num_threads=num_threads_cpu_resampling,
                                              device=torch.device('cpu'))[0].numpy()
                save_res.append(tps.starmap_async(
                    _saver, ((c, seg_resampled, ts, td, to),)
                ))
                break
            if i == len(load_res) - 1:
                sleep(0.1)
    tp.close()
    tp.join()

    _ = [i.get() for i in save_res]
    tps.close()
    tps.join()

    # for f in files:
    #     if overwrite or not isfile(join(target_folder, f)):
    #         print(os.path.basename(f))
    #         seg = sitk.GetArrayFromImage(sitk.ReadImage(join(pred_folder, f))).astype(np.uint8)
    #         target_file = join(ref_folder, f)
    #         if not isfile(target_file):
    #             # when the target file is an image it will have the _0000 suffix
    #             target_file = target_file[:-7] + '_0000.nii.gz'
    #         target_itk_img = sitk.ReadImage(target_file)
    #         target_shape = sitk.GetArrayFromImage(target_itk_img).shape
    #         try:
    #             seg_resampled = \
    #                 resample_torch_simple(torch.from_numpy(seg)[None], target_shape, is_seg=True,
    #                                num_threads=num_threads,
    #                                device=torch.device('cuda:0'))[0].numpy()
    #         except:
    #             print(f'CPU, file: {f}')
    #             seg_resampled = \
    #                 resample_torch_simple(torch.from_numpy(seg)[None], target_shape, is_seg=True,
    #                                num_threads=num_threads,
    #                                device=torch.device('cpu'))[0].numpy()
    #         torch.cuda.empty_cache()
    #         seg_resampled_itk = sitk.GetImageFromArray(seg_resampled)
    #         seg_resampled_itk.CopyInformation(target_itk_img)
    #         sitk.WriteImage(seg_resampled_itk, join(target_folder, f))


if __name__ == '__main__':
    set_start_method('spawn')

    import argparse
    parser = argparse.ArgumentParser("This script takes a folder containing segmentation nifti files and resizes "
                                     "them to corresponding (equally named) niftis on the ref folder. The niftis in ref "
                                     "folder can contain anything (doesn't matter if its images or segs). "
                                     "\nTHIS WILL BE RUN ON GPU! There is a CPU fallback in case GPU memory is "
                                     "insufficient (see -np)")
    parser.add_argument('-i', type=str, required=True,
                        help='Input folder. Must contain nifti files with segmentations')
    parser.add_argument('-o', type=str, required=True,
                        help="Output folder. Must be empty! If it doesn't exist it will be created")
    parser.add_argument('-ref', type=str, required=True,
                        help="Reference folder. Must contain files with the same name as the segmentations in -i. "
                             "Segmentations will be reshaped to their counterparts in -ref")
    parser.add_argument('-np', type=int, required=False, default=8,
                        help='Number of threads used for resampling on CPU fallback (if GPU fails). Default: 8. '
                             'Make this a lot higher if you can!')
    parser.add_argument('--overwrite_existing', action='store_true',
                        help='By default the script will skip existing results. Set this flag to overwrite (recompute) '
                             'them instead.')
    args = parser.parse_args()

    resample_segmentations_to_ref(args.ref, args.i, args.o, overwrite=args.overwrite_existing, num_threads_cpu_resampling=args.np)

    # folders_pred = [
    #     # semantic segs
    #
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    #
    #     # instance segs
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
    # ]
    #
    # original_seg_folder_tr = '/dkfz/cluster/gpu/data/OE0441/isensee/Shank_testSet/original_segs/labelsTr'
    # for f in folders_pred:
    #     resample_segmentations_to_ref(original_seg_folder_tr, f, f + '_resized', overwrite=False, num_threads=128)

