from multiprocessing import Pool
import numpy as np
import SimpleITK as sitk
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.sitk_stuff import copy_geometry


def assign_correct_tooth_labels_to_instanceseg(semseg_image: str, instanceseg_image: str, output_filename: str,
                                               min_tooth_volume: float,
                                               isolated_semsegs_as_new_instances: bool = False,
                                               overwrite: bool = True,
                                               allow_background_label: bool = False) -> None:
    if overwrite or not isfile(output_filename):
        try:
            semseg_itk = sitk.ReadImage(semseg_image)
            instanceseg_itk = sitk.ReadImage(instanceseg_image)
            semseg_npy = sitk.GetArrayFromImage(semseg_itk).astype(np.uint8)
            instanceseg_npy = sitk.GetArrayFromImage(instanceseg_itk).astype(np.uint8)

            output = np.zeros_like(semseg_npy)

            instances = np.sort(pd.unique(instanceseg_npy.ravel()))
            for i in instances:
                if i == 0: continue
                instance_mask = instanceseg_npy == i

                seg_predictions = semseg_npy[instance_mask]

                freq = np.bincount(seg_predictions)
                if len(freq) == 1:
                    # no semantic segmentation in that instance
                    continue
                if allow_background_label:
                    predicted_label = np.argmax(freq)
                else:
                    predicted_label = np.argmax(freq[1:]) + 1

                output[instance_mask] = predicted_label

            if isolated_semsegs_as_new_instances:
                labels = [i for i in np.sort(pd.unique(semseg_npy.ravel())) if i != 0]
                for l in labels:
                    mask = semseg_npy == l
                    if np.sum(instanceseg_npy[mask]) == 0:
                        output[mask] = l

            if min_tooth_volume > 0:
                vol_per_voxel = np.prod(semseg_itk.GetSpacing())
                n_pixel_cutoff = min_tooth_volume / vol_per_voxel
                instances = [i for i in pd.unique(output.ravel()) if i != 0]
                for i in instances:
                    mask = output == i
                    if np.sum(mask) < n_pixel_cutoff:
                        output[mask] = 0

            output_itk = sitk.GetImageFromArray(output)
            output_itk = copy_geometry(output_itk, semseg_itk)
            sitk.WriteImage(output_itk, output_filename)
        except Exception as e:
            print(f'error in {semseg_image, instanceseg_image, output_filename}')
            raise e


def assign_tooth_labels_to_all_instancesegs(semseg_folder, instanceseg_folder, output_folder, min_tooth_volume: float = 3,
                                            isolated_semsegs_as_new_instances: bool = False,
                                            num_processes: int = 12, overwrite: bool = True,
                                            allow_bg: bool = False):
    p = Pool(num_processes)
    maybe_mkdir_p(output_folder)

    files = nifti_files(instanceseg_folder, join=False)
    assert all([i in nifti_files(semseg_folder, join=False) for i in files])
    # assign_correct_tooth_labels_to_instanceseg(*list(        zip(
    #         [join(semseg_folder, i) for i in files],
    #         [join(instanceseg_folder, i) for i in files],
    #         [join(output_folder, i) for i in files],
    #         [min_instance_size] * len(files),
    #         [isolated_semsegs_as_new_instances] * len(files),
    #         [overwrite] * len(files),
    #     ))[0])

    # create clean tooth instances
    r = p.starmap_async(
        assign_correct_tooth_labels_to_instanceseg,
        zip(
            [join(semseg_folder, i) for i in files],
            [join(instanceseg_folder, i) for i in files],
            [join(output_folder, i) for i in files],
            [min_tooth_volume] * len(files),
            [isolated_semsegs_as_new_instances] * len(files),
            [overwrite] * len(files),
            [allow_bg] * len(files),
        )
    )
    r.get()
    p.close()
    p.join()


if __name__ == '__main__':
    folder_semseg = '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8/fold_0/validation_resized'

    folder_raw_instances = '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances_resized'
    folder_out = folder_raw_instances + '_181_sp03_256_bs8_mintoothsize20_isoFalse'
    assign_tooth_labels_to_all_instancesegs(folder_semseg, folder_raw_instances, folder_out, 20, isolated_semsegs_as_new_instances=False, num_processes=190, overwrite=False)

    folder_raw_instances = '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_256_bs8/fold_0/validation_instances_resized'
    folder_out = folder_raw_instances + '_181_sp03_256_bs8_mintoothsize20_isoFalse'
    assign_tooth_labels_to_all_instancesegs(folder_semseg, folder_raw_instances, folder_out, 20, isolated_semsegs_as_new_instances=False, num_processes=190, overwrite=False)

    folder_raw_instances = '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs16/fold_0/validation_instances_resized'
    folder_out = folder_raw_instances + '_181_sp03_256_bs8_mintoothsize20_isoFalse'
    assign_tooth_labels_to_all_instancesegs(folder_semseg, folder_raw_instances, folder_out, 20, isolated_semsegs_as_new_instances=False, num_processes=190, overwrite=False)

