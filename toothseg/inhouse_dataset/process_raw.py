import glob
import os
import shutil
from multiprocessing import Pool

import pydicom
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from toothseg.inhouse_dataset.utils import (
    psg_to_npa,
    convert_label_ISO_to_continious,
    convert_ISO_to_continious,
    copy_geometry,
)


def convert_case(input_folder: str, output_dir: str):
    """
    Convert the .dcm data into nii.gz format and convert labels into an ascendind continuous format

    :param input_folder: folder containing .dcm files
    :param output_dir: directory in which the .nii.gz will be saved in
    :return:
    """
    try:
        # we use this order to determine what is allowed to overwrite what
        subfolder_order = [
            "LOWER_JAW",
            "UPPER_JAW",
            "TOOTH",
            "SUPERNUMERARY_TOOTH",
            "PRIMARY_TOOTH",
            "NON_TOOTH_SUPPORTED_CROWN",
            "DENTAL_IMPLANT",
            "METAL_FILLING",
            "METAL_CROWN",
        ]

        scan_folder = join(input_folder, "scan")
        dcm_files = subfiles(scan_folder, suffix=".dcm")
        assert len(dcm_files) == 1, f"Scan file not found for patient {input_folder}"
        ds = pydicom.read_file(dcm_files[0])
        if hasattr(ds, "PixelSpacing"):
            in_plane = [float(i) for i in ds.PixelSpacing]
        else:
            in_plane = None
        if hasattr(ds, "SliceThickness"):
            slice_thickness = float(ds.SliceThickness)
        else:
            slice_thickness = None
        if in_plane is not None and slice_thickness is not None:
            spacing = [slice_thickness] + in_plane
            image = ds.pixel_array
        else:
            print(input_folder, "fallback to itk")
            img = sitk.ReadImage(dcm_files[0])
            spacing = list(img.GetSpacing())[::-1]
            image = sitk.GetArrayFromImage(img)

        seg = np.zeros(image.shape)
        psg_folder = join(input_folder, "psg_manual_ann")
        psg_subfolders = subfolders(psg_folder, join=False)
        remaining_subfolders = [i for i in subfolder_order if i in psg_subfolders]
        # now append additional folders
        remaining_subfolders += [i for i in psg_subfolders if i not in remaining_subfolders]

        for psg_fld in remaining_subfolders:
            psg_files = subfiles(join(psg_folder, psg_fld), suffix=".psg")
            for psg_file in psg_files:
                seg_here, label_text, label_int = psg_to_npa(psg_file)
                label_here = convert_label_ISO_to_continious(label_text, label_int)
                # for real?
                seg_here = seg_here.transpose((2, 0, 1))[::-1]
                # we ignore overwriting jaws.
                if np.sum(seg[seg_here != 0] > 2) > 100:
                    overwritten = np.unique(seg[seg_here != 0])
                    print(
                        f" {input_folder} \noverwriting labels"
                        f" {overwritten}\nnum_pixels{np.sum(seg[seg_here != 0])}\nwith label"
                        f" {label_here}"
                    )
                seg[seg_here != 0] = label_here

        file_name = os.path.split(input_folder)[-1]

        image_itk = sitk.GetImageFromArray(image)
        image_itk.SetSpacing(spacing[::-1])
        sitk.WriteImage(image_itk, join(output_dir, "imagesTr", file_name + ".nii.gz"))
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk.SetSpacing(spacing[::-1])
        sitk.WriteImage(seg_itk, join(output_dir, "labelsTr", file_name + ".nii.gz"))
        print(input_folder, image.shape, list(spacing), "\n")
    except Exception as e:
        print(e, "\n", input_folder)


def copy_information_from_raw_to_new_labels(input_dir, changed_dir):
    """
    The new labels are given in the same format as the raw data but only contains the changes.
    --> copy the rest from the raw data to complete the new labels folders

    :param input_dir: folder
    :param changed_dir:
    :return:
    """
    subfolders = [
        "LOWER_JAW",
        "UPPER_JAW",
        "SUPERNUMERARY_TOOTH",
        "PRIMARY_TOOTH",
        "NON_TOOTH_SUPPORTED_CROWN",
        "DENTAL_IMPLANT",
        "METAL_FILLING",
        "METAL_CROWN",
    ]
    names = os.listdir(changed_dir)
    print(f"Copy Information for {len(names)} files")

    for name in tqdm(names):
        # Copy image information (in scan folder)
        shutil.copytree(
            os.path.join(input_dir, name, "scan"),
            os.path.join(changed_dir, name, "scan"),
            dirs_exist_ok=True,
        )
        # Copy subfolders
        for subfolder in subfolders:
            if os.path.exists(os.path.join(input_dir, name, "psg_manual_ann", subfolder)):
                shutil.copytree(
                    os.path.join(input_dir, name, "psg_manual_ann", subfolder),
                    os.path.join(changed_dir, name, "psg_manual_ann", subfolder),
                    dirs_exist_ok=True,
                )


def update_with_new_segmentation(input_file, target_folder):
    """
    The new segmentation only contains the Teeth classes. Load the old segmentation and
    update the Tooth classes with the new ones but keep the other classes.

    :param input_file: file which contains the updated segmentation
    :param target_folder: output folder which contains the old segmentations which gets updated
    :return:
    """
    file_name = os.path.split(input_file)[-1]
    labels = sitk.ReadImage(input_file)
    new_labels = convert_ISO_to_continious(sitk.GetArrayFromImage(labels))

    lables_old = sitk.GetArrayFromImage(sitk.ReadImage(join(target_folder, "labelsTr", file_name)))

    x, y, z = np.where(
        (
            (5 > lables_old) | (lables_old > 56)
        )  # only use classes outside the TOOTH classes (Jaw etc)
        & (lables_old != 0)  # ignore background
        & (new_labels == 0)
    )  # dont overwrite changes

    new_labels[x, y, z] = lables_old[x, y, z]
    new_labels = new_labels.astype(np.uint8)

    output_itk = sitk.GetImageFromArray(new_labels)
    output_itk = copy_geometry(output_itk, labels)
    sitk.WriteImage(output_itk, join(target_folder, "labelsTr", file_name))


if __name__ == "__main__":
    dataset_name = "Dataset164_All_Classes"

    root_raw = "/home/l727r/Documents/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/RadboudUni_2022_ShankCBCTTeeth/Raw_Database"
    output_processed = "/media/l727r/data/Teeth_Data/Processed_Database"
    output_processed = join(output_processed, dataset_name)

    txt_file = "ignore_files.txt"
    with open(txt_file, "r") as file:
        ignore_files = [line.strip() for line in file]
    """
    1. Collect Data
    Collect all Image Files in the Raw Database which should be processed.
    A few of them have to be excluded due wrong annotations or wrong geometrie.
    All files to be ignored are in listed in ignore_files.txt.
    """
    cases = subfolders(join(root_raw, "raw_data"))
    print(f"Total number of Cases: {len(cases)}")
    cases = [case for case in cases if os.path.split(case)[-1] not in ignore_files]
    cases.sort()
    print(f"Number of Cases after Filtering: {len(cases)}")

    """
    2. Process Data 
    Convert the Data from .dcm to .nii.gz.
    Convert Teeth Labels from ISO numbering format inta a continuous ascending format 
    https://en.wikipedia.org/wiki/Dental_notation
    LOWER_JAW -> 1
    UPPER_JAW -> 2
    DENTAL_IMPLANT -> 3
    NON_TOOTH_SUPPORTED_CROWN -> 4
    [11 - 18] -> [5 - 12]
    [21 - 28] -> [13 - 20]
    [31 - 38] -> [21 - 28]
    [41 - 48] -> [29 - 36]
    """
    os.makedirs(output_processed, exist_ok=True)
    os.makedirs(join(output_processed, "imagesTr"), exist_ok=True)
    os.makedirs(join(output_processed, "labelsTr"), exist_ok=True)

    p = Pool(16)
    async_results = []
    for case in cases:
        async_results.append(
            p.starmap_async(
                convert_case,
                (
                    (
                        case,
                        output_processed,
                    ),
                ),
            )
        )

    _ = [a.get() for a in tqdm(async_results, desc="Processing...")]
    p.close()
    p.join()

    """
    3. Incorporate Updates
    Some Labels got corrected and this corrections have to be included into the Processed Database
    - new_labels/: some Tooth have the wrong label and have to be corrected (given in old format)
                   new_labels only contain the changed labels --> rest has to be collected from raw_data
    - new_segmentations/: complete corrected segmentation but only for teeth (given in old format)
                   other classes (Jaw etc.) have to be collected from raw.
    - new_segmentations_correct_format/: complete corrected segmentation including all classes in the 
                   correct format --> can be directly copied.     
    """

    """
    3.1 New Labels
    Copy missing information from raw_data to root_changes/output since only the changed data is new_labels
    """
    # copy_information_from_raw_to_new_labels(join(root_raw, "raw_data"), join(root_raw, "new_labels"))
    cases = subfolders(join(root_raw, "new_labels"))
    print(f"Total number of Cases with new labels: {len(cases)}")
    cases = [case for case in cases if os.path.split(case)[-1] not in ignore_files]
    cases.sort()
    print(f"Number of Cases with new labels after Filtering: {len(cases)}")

    p = Pool(16)
    async_results = []
    for case in cases:
        async_results.append(
            p.starmap_async(
                convert_case,
                (
                    (
                        case,
                        output_processed,
                    ),
                ),
            )
        )

    _ = [a.get() for a in tqdm(async_results, desc="Processing...")]
    p.close()
    p.join()

    """
    3.2 New Segmentations in the old format
    Only Teeth Classes are included into the new Segmentations
    The other classes like Jar etc. have to be collected form the old segmentation
    """
    cases = glob.glob(join(root_raw, "new_segmentations", "*.nii.gz"))
    print(f"Total number of Cases with new segmentations: {len(cases)}")
    cases = [
        case for case in cases if os.path.split(case)[-1].replace(".nii.gz", "") not in ignore_files
    ]
    print(f"Number of Cases with new segmentations after Filtering: {len(cases)}")

    cases.sort()
    p = Pool(16)
    async_results = []
    for case in cases:
        async_results.append(
            p.starmap_async(
                update_with_new_segmentation,
                (
                    (
                        case,
                        output_processed,
                    ),
                ),
            )
        )

    _ = [a.get() for a in tqdm(async_results, desc="Processing...")]
    p.close()
    p.join()

    """
    3.3 New Segmentations in the correct Format
    New segmentations in the correct format and also with all classes
    """
    cases = glob.glob(join(root_raw, "new_segmentations_correct_format", "*.nii.gz"))
    print(f"Total number of Cases with new segmentations: {len(cases)}")
    cases = [
        case for case in cases if os.path.split(case)[-1].replace(".nii.gz", "") not in ignore_files
    ]
    cases.sort()
    print(f"Number of Cases with new segmentations after Filtering: {len(cases)}")

    for case in cases:
        file = os.path.split(case)[-1]
        print(file)
        shutil.copy(case, join(output_processed, "labelsTr", file))

    """
    4. Generate dataset.json
    """
    label_dict = {
        "background": 0,
        "Lower Jaw": 1,
        "Upper Jaw": 2,
        "Dental Implant": 3,
        "Non tooth supported crown": 4,
    }
    # Permanent Teeth
    label_dict.update({10 + i: 4 + i for i in range(1, 9)})  # [11 - 18] -> [5 - 12]
    label_dict.update({20 + i: 12 + i for i in range(1, 9)})  # [21 - 28] -> [13 - 20]
    label_dict.update({30 + i: 20 + i for i in range(1, 9)})  # [31 - 38] -> [21 - 28]
    label_dict.update({40 + i: 28 + i for i in range(1, 9)})  # [41 - 48] -> [29 - 36]
    # Primary Teeth
    label_dict.update({50 + i: 36 + i for i in range(1, 6)})  # [51 - 55] -> [37 - 41]
    label_dict.update({60 + i: 41 + i for i in range(1, 6)})  # [61 - 65] -> [42 - 16]
    label_dict.update({70 + i: 46 + i for i in range(1, 6)})  # [71 - 75] -> [47 - 51]
    label_dict.update({80 + i: 51 + i for i in range(1, 6)})  # [81 - 85] -> [52 - 56]

    generate_dataset_json(
        output_folder=output_processed,
        channel_names={0: "CT"},
        labels=label_dict,
        num_training_cases=len(glob.glob(join(output_processed, "imagesTr", "*.nii.gz"))),
        file_ending=".nii.gz",
        dataset_name=dataset_name,
    )
