import glob
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm

from toothseg.datasets.inhouse_dataset.utils import copy_geometry


def process_label(input_label, input_img, output_label, output_img):
    """
    Filtering the labels to only contain the Permanent Teeth, files with primary teeth are ignored.
    1-4: Lower/Upper Jaw; Implants/ Crow -> 0
    5-36: Permanent Teeth Classes -> reduce by -4
    >36: Primary Teeth -> exclude file
    :param input_label: label file
    :param input_img: image file
    :param output_label: output location of label file
    :param output_img: output location of image file
    :return:
    """

    labels_itk = sitk.ReadImage(input_label)
    labels = sitk.GetArrayFromImage(labels_itk)
    if np.any(labels > 36):
        print(f"Primary Teeth in:{input_label}")
        return
    labels[labels < 5] = 0
    labels[labels > 0] -= 4
    labels = labels.astype(np.uint8)

    output_itk = sitk.GetImageFromArray(labels)
    output_itk = copy_geometry(output_itk, labels_itk)
    sitk.WriteImage(output_itk, output_label)
    shutil.copy(input_img, output_img)


if __name__ == "__main__":
    dataset_name = "Dataset164_Filtered_Classes"

    input_dir = "/media/l727r/data/Teeth_Data/Processed_Database/Dataset164_All_Classes"
    output_dir = "/media/l727r/data/Teeth_Data/Processed_Database/"
    output_dir = join(output_dir, dataset_name)

    cases = os.listdir(join(input_dir, "labelsTr"))
    print(f"in Total {len(cases)} Files are found")
    os.makedirs(join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(join(output_dir, "imagesTs"), exist_ok=True)
    os.makedirs(join(output_dir, "labelsTr"), exist_ok=True)
    os.makedirs(join(output_dir, "labelsTs"), exist_ok=True)

    txt_file = "train.txt"
    with open(txt_file, "r") as file:
        train_files = [line.strip() for line in file]

    txt_file = "test.txt"
    with open(txt_file, "r") as file:
        test_files = [line.strip() for line in file]

    """
    Filte the labels to only contain Permanent Teeth.
    Ignore Files which include Primary Teeth
    Split into Train and Test set
    """
    p = Pool(16)
    async_results = []
    for case in cases:
        if case.replace(".nii.gz", "") in train_files:
            folder = "Tr"
        elif case.replace(".nii.gz", "") in test_files:
            folder = "Ts"
        else:
            print(f"File {case} not found in Train or Test Split")
            continue

        async_results.append(
            p.starmap_async(
                process_label,
                (
                    (
                        join(input_dir, "labelsTr", case),
                        join(input_dir, "imagesTr", case),
                        join(output_dir, "labels" + folder, case),
                        join(
                            output_dir, "images" + folder, case.replace(".nii.gz", "_0000.nii.gz")
                        ),
                    ),
                ),
            )
        )

    _ = [a.get() for a in tqdm(async_results, desc="Processing...")]
    p.close()
    p.join()

    """
    Generate dataset.json
    """
    label_dict = {
        "background": 0,
    }
    # Permanent Teeth
    label_dict.update({10 + i: i for i in range(1, 9)})  # [11 - 18] -> [1 - 8]
    label_dict.update({20 + i: 8 + i for i in range(1, 9)})  # [21 - 28] -> [9 - 16]
    label_dict.update({30 + i: 16 + i for i in range(1, 9)})  # [31 - 38] -> [17 - 24]
    label_dict.update({40 + i: 24 + i for i in range(1, 9)})  # [41 - 48] -> [25 - 32]

    generate_dataset_json(
        output_folder=output_dir,
        channel_names={0: "CT"},
        labels=label_dict,
        num_training_cases=len(glob.glob(join(output_dir, "imagesTr", "*.nii.gz"))),
        numTest=len(glob.glob(join(output_dir, "imagesTs", "*.nii.gz"))),
        file_ending=".nii.gz",
        dataset_name=dataset_name,
    )
