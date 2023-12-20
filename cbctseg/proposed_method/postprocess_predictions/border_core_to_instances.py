from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg, \
    postprocess_instance_segmentation
from nnunet.utilities.sitk_stuff import copy_geometry


def convert_all_sem_to_instance(border_core_seg_folder, output_folder, small_center_threshold=0.03,
                                isolated_border_as_separate_instance_threshold=0.03, num_processes: int = 12,
                                overwrite: bool = True):
    maybe_mkdir_p(output_folder)

    input_files = nifti_files(border_core_seg_folder, join=False)
    output_files = [join(output_folder, i) for i in input_files]
    input_files = [join(border_core_seg_folder, i) for i in input_files]

    p = Pool(num_processes)
    res = p.starmap_async(
        load_convert_semantic_to_instance_save,
        zip(input_files,
            output_files,
            [small_center_threshold] * len(output_files),
            [isolated_border_as_separate_instance_threshold] * len(output_files),
            [overwrite] * len(output_files)
            )
    )
    _ = res.get()
    p.close()
    p.join()


def load_convert_semantic_to_instance_save(input_file: str, output_file: str, small_center_threshold=0.03,
                                           isolated_border_as_separate_instance_threshold=0.03, overwrite=True):
    if overwrite or not isfile(output_file):
        print(os.path.basename(input_file))
        itk_img = sitk.ReadImage(input_file)
        npy_img = sitk.GetArrayFromImage(itk_img)
        spacing = np.array(itk_img.GetSpacing())[::-1]
        instance_seg = convert_semantic_to_instanceseg(npy_img, spacing, small_center_threshold,
                                                       isolated_border_as_separate_instance_threshold)
        instance_seg = postprocess_instance_segmentation(instance_seg)
        itk_res = sitk.GetImageFromArray(instance_seg)
        itk_res = copy_geometry(itk_res, itk_img)
        sitk.WriteImage(itk_res, output_file)


if __name__ == '__main__':

    folders_pred = [
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
    ]
    small_center_threshold = 0.3**3 * 10  # equivalent to 10 pixels at 0.3 spacing
    isolated_border_as_separate_instance_threshold = 0.3**3 * 10  # equivalent to 10 pixels at 0.3 spacing
    num_processes = 64
    for f in folders_pred:
        convert_all_sem_to_instance(f, f + '_instances', overwrite=False,
                                    small_center_threshold=small_center_threshold,
                                    isolated_border_as_separate_instance_threshold=
                                    isolated_border_as_separate_instance_threshold,
                                    num_processes=num_processes)
