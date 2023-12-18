import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch


def resample_segmentations_to_ref(ref_folder, pred_folder, target_folder, overwrite=False, num_threads=128):
    maybe_mkdir_p(target_folder)
    files = nifti_files(pred_folder, join=False)
    for f in files:
        if overwrite or not isfile(join(target_folder, f)):
            print(os.path.basename(f))
            seg = sitk.GetArrayFromImage(sitk.ReadImage(join(pred_folder, f))).astype(np.uint8)
            target_itk_img = sitk.ReadImage(join(ref_folder, f))
            target_shape = sitk.GetArrayFromImage(target_itk_img).shape
            try:
                seg_resampled = \
                    resample_torch(torch.from_numpy(seg)[None], target_shape, None, None, is_seg=True,
                                   num_threads=num_threads,
                                   device=torch.device('cuda:0'))[0].numpy()
            except:
                print(f'CPU, file: {f}')
                seg_resampled = \
                    resample_torch(torch.from_numpy(seg)[None], target_shape, None, None, is_seg=True,
                                   num_threads=num_threads,
                                   device=torch.device('cpu'))[0].numpy()
            torch.cuda.empty_cache()
            seg_resampled_itk = sitk.GetImageFromArray(seg_resampled)
            seg_resampled_itk.CopyInformation(target_itk_img)
            sitk.WriteImage(seg_resampled_itk, join(target_folder, f))


if __name__ == '__main__':
    folders_pred = [
        # semantic segs

        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation',

        # instance segs
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances',
        '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances',
        # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances',
    ]

    original_seg_folder_tr = '/dkfz/cluster/gpu/data/OE0441/isensee/Shank_testSet/original_segs/labelsTr'
    for f in folders_pred:
        resample_segmentations_to_ref(original_seg_folder_tr, f, f + '_resized', overwrite=False, num_threads=128)

