from multiprocessing import Pool
import numpy as np
import SimpleITK as sitk
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.sitk_stuff import copy_geometry


def assign_correct_tooth_labels_to_instanceseg(semseg_image: str, instanceseg_image: str, output_filename: str,
                                               min_instance_size: float,
                                               isolated_semsegs_as_new_instances: bool = False,
                                               overwrite: bool = True) -> None:
    if overwrite or isfile(output_filename):
        try:
            semseg_itk = sitk.ReadImage(semseg_image)
            instanceseg_itk = sitk.ReadImage(instanceseg_image)
            semseg_npy = sitk.GetArrayFromImage(semseg_itk).astype(np.uint8)
            instanceseg_npy = sitk.GetArrayFromImage(instanceseg_itk).astype(np.uint8)

            vol_per_voxel = np.prod(semseg_itk.GetSpacing())
            n_pixel_cutoff = min_instance_size / vol_per_voxel

            output = np.zeros_like(semseg_npy)

            instances = np.sort(pd.unique(instanceseg_npy.ravel()))
            for i in instances:
                if i == 0: continue
                instance_mask = instanceseg_npy == i
                if np.sum(instance_mask) < n_pixel_cutoff:
                    continue
                seg_predictions = semseg_npy[instance_mask]

                freq = np.bincount(seg_predictions)
                if len(freq) == 1:
                    # no semantic segmentation in that instance
                    continue
                # [1:] because we dont want something to end up as 0? TODO evaluate this?
                predicted_label = np.argmax(freq[1:]) + 1

                output[instance_mask] = predicted_label

            if isolated_semsegs_as_new_instances:
                labels = [i for i in np.sort(pd.unique(semseg_npy.ravel())) if i != 0]
                for l in labels:
                    mask = semseg_npy == l
                    if np.sum(instanceseg_npy[mask]) == 0:
                        output[mask] = l

            output_itk = sitk.GetImageFromArray(output)
            output_itk = copy_geometry(output_itk, semseg_itk)
            sitk.WriteImage(output_itk, output_filename)
        except Exception as e:
            print(f'error in {semseg_image, instanceseg_image, output_filename}')
            raise e


def assign_tooth_labels_to_all_instancesegs(semseg_folder, instanceseg_folder, output_folder, min_instance_size: float = 3,
                                            isolated_semsegs_as_new_instances: bool = False,
                                            num_processes: int = 12, overwrite: bool = True):
    p = Pool(num_processes)
    maybe_mkdir_p(output_folder)

    files = nifti_files(instanceseg_folder, join=False)
    assert all([i in nifti_files(semseg_folder, join=False) for i in files])

    # create clean tooth instances
    r = p.starmap_async(
        assign_correct_tooth_labels_to_instanceseg,
        zip(
            [join(semseg_folder, i) for i in files],
            [join(instanceseg_folder, i) for i in files],
            [join(output_folder, i) for i in files],
            [min_instance_size] * len(files),
            [isolated_semsegs_as_new_instances] * len(files),
            [overwrite] * len(files),
        )
    )
    r.get()
    p.close()
    p.join()