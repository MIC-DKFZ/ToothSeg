from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.sitk_stuff import copy_geometry


def convert_shanks_labels_to_mine(label_text: str, label_int: str):
    if label_text == 'LOWER_JAW':
        label_int = 1
    elif label_text == 'UPPER_JAW':
        label_int = 2
    elif label_text == 'DENTAL_IMPLANT':
        label_int = 3
    elif label_text == 'NON_TOOTH_SUPPORTED_CROWN':
        label_int = 4
    else:
        # up to 8 teeth per row. oof
        assert label_int != ''
        dct = {10 + i: 4 + i for i in range(1, 9)}  # [11 - 18] -> [5 - 12]
        dct.update({20 + i: 12 + i for i in range(1, 9)})  # [21 - 28] -> [13 - 20]
        dct.update({30 + i: 20 + i for i in range(1, 9)})  # [31 - 38] -> [21 - 28]
        dct.update({40 + i: 28 + i for i in range(1, 9)})  # [41 - 48] -> [29 - 36]

        # in addition some kids got milk teeth. Oof. up to 5 more per row
        dct.update({50 + i: 36 + i for i in range(1, 6)})
        dct.update({60 + i: 41 + i for i in range(1, 6)})
        dct.update({70 + i: 46 + i for i in range(1, 6)})
        dct.update({80 + i: 51 + i for i in range(1, 6)})

        label_int = dct[int(label_int)]
    return label_int


def convert_adult_teeth_segmentation_back_to_shanks_labels(predicted_image: np.ndarray) -> np.ndarray:
    output = np.zeros_like(predicted_image, dtype=np.uint8)
    mask = (predicted_image > 4) & (predicted_image < 13)
    output[mask] = predicted_image[mask] + (11 - 5)
    mask = (predicted_image > 12) & (predicted_image < 21)
    output[mask] = predicted_image[mask] + (21 - 13)
    mask = (predicted_image > 20) & (predicted_image < 29)
    output[mask] = predicted_image[mask] + (31 - 21)
    mask = (predicted_image > 28) & (predicted_image < 37)
    output[mask] = predicted_image[mask] + (41 - 29)
    return output


def convert_label_id_back_to_shanks_convention(label_id: int) -> int:
    if 4 < label_id < 13:
        return label_id + (11 - 5)
    if 12 < label_id < 21:
        return label_id + (21 - 13)
    if 20 < label_id < 29:
        return label_id + (31 - 21)
    if 28 < label_id < 37:
        return label_id + (41 - 29)
    raise RuntimeError(f'Unexpected label {label_id}')


def convert_single_prediction_adult_teeth_back_to_shanks_labels(input_file: str, output_file: str):
    input_itk = sitk.ReadImage(input_file)
    input_npy = sitk.GetArrayFromImage(input_itk)
    output_npy = convert_adult_teeth_segmentation_back_to_shanks_labels(input_npy)
    output_itk = sitk.GetImageFromArray(output_npy)
    output_itk = copy_geometry(output_itk, input_itk)
    sitk.WriteImage(output_itk, output_file)


def convert_all_adult_teeth_back_to_shanks_labels(source_folder: str, target_folder: str, num_processes: int = 8):
    maybe_mkdir_p(target_folder)
    source_niftis = nifti_files(source_folder, join=False)
    target_niftis = [join(target_folder, i) for i in source_niftis]
    source_niftis = [join(source_folder, i) for i in source_niftis]

    p = Pool(num_processes)
    r = p.starmap_async(
        convert_single_prediction_adult_teeth_back_to_shanks_labels,
        zip(source_niftis, target_niftis)
    )
    _ = r.get()
    p.close()
    p.join()


if __name__ == '__main__':
    convert_all_adult_teeth_back_to_shanks_labels(
        '/home/isensee/drives/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/RadboudUni_2022_ShankCBCTTeeth/2022_05_13_Predictions/gt',
        '/home/isensee/drives/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/RadboudUni_2022_ShankCBCTTeeth/2022_05_13_Predictions/gt_ShanksLabels')
    convert_all_adult_teeth_back_to_shanks_labels(
        '/home/isensee/drives/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/RadboudUni_2022_ShankCBCTTeeth/2022_05_13_Predictions/Task168__nnUNetTrainerV2_onlySomeMirroring2_DiceCE_noSmooth__nnUNetPlansv2.1_trgSp_05x05x05_bs5_convertedToInstance_correctClass',
        '/home/isensee/drives/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/RadboudUni_2022_ShankCBCTTeeth/2022_05_13_Predictions/Task168__nnUNetTrainerV2_onlySomeMirroring2_DiceCE_noSmooth__nnUNetPlansv2.1_trgSp_05x05x05_bs5_convertedToInstance_correctClass_ShanksLabels')
