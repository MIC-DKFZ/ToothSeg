import glob
import os
from os.path import join

import numpy as np
from tqdm import tqdm
import shutil
from multiprocessing import Pool
from convert_to_nifti import convert_case
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib
import pandas as pd


def copy_information_from_raw_to_changed(input_dir, changed_dir):
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
        # Copy image information (in scane folder)
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


def generate_img_and_seg(base):
    p = Pool(16)
    cases = subfolders(base)
    print(f"Generate Images and Sementations for {len(cases)} Files")
    r = p.map_async(convert_case, cases)
    p.close()
    p.join()


def copy_updated_labels(root_input, root_output):
    folders = os.listdir(root_input)
    for folder in tqdm(folders):
        shutil.copy(
            join(root_input, folder, "image.nii.gz"),
            join(root_output, "imagesTr_labels_updated", folder + "_0000.nii.gz"),
        )
        shutil.copy(
            join(root_input, folder, "seg.nii.gz"),
            join(root_output, "labelsTr_labels_updated", folder + ".nii.gz"),
        )
        # quit()


def convert_to_Fabi(img):
    dct = {10 + i: 4 + i for i in range(1, 9)}  # [11 - 18] -> [5 - 12]
    dct.update({20 + i: 12 + i for i in range(1, 9)})  # [21 - 28] -> [13 - 20]
    dct.update({30 + i: 20 + i for i in range(1, 9)})  # [31 - 38] -> [21 - 28]
    dct.update({40 + i: 28 + i for i in range(1, 9)})  # [41 - 48] -> [29 - 36]

    # in addition some kids got milk teeth. Oof. up to 5 more per row
    dct.update({50 + i: 36 + i for i in range(1, 6)})
    dct.update({60 + i: 41 + i for i in range(1, 6)})
    dct.update({70 + i: 46 + i for i in range(1, 6)})
    dct.update({80 + i: 51 + i for i in range(1, 6)})

    new_labels = np.zeros(img.shape, dtype=np.uint8)
    for val in np.unique(img):
        if val == 0:
            continue
        new_val = dct[int(val)]
        x, y, z = np.where(img == val)
        new_labels[x, y, z] = new_val
    return new_labels

def process_one_file(file,root_changes,root_output,root_input):
    print(file)
    labels = nib.load(join(root_changes, file))
    print(labels.shape)
    new_labels = convert_to_Fabi(labels.get_fdata())
    print("Conv",new_labels.shape)
    lables_old = nib.load(join(root_input, file.replace(".nii.gz",""), "seg.nii.gz")).get_fdata()
    print(lables_old.shape)
    x, y, z = np.where(((5 > lables_old)|(lables_old>56))   # only use classes outside the TOOTH classes (Jaw etc)
                       &(lables_old!=0)                     # ignore background
                       &(new_labels==0))                    # dont overwrite changes

    new_labels[x, y, z] = lables_old[x, y, z]

    labels_nib = nib.Nifti1Image(new_labels, labels.affine)
    nib.save(labels_nib, join(root_output, "labelsTs_seg_updated", file))

    shutil.copy(
        join(root_input, file.replace(".nii.gz",""),"image.nii.gz"),
        join(root_output, "imagesTs_seg_updated", file).replace(".nii.gz", "_0000.nii.gz"),
    )

def convert_seg_to_Fabi(root_changes, root_output, root_input):

    files = os.listdir(root_changes)
    print(f"Convert Sementations for {len(files)} Files")

    p = Pool(8)
    res=[]
    for file in tqdm(files):
        res.append(
            p.starmap_async(
                process_one_file,
                ((file,root_changes, root_output, root_input),),
            )
        )

    _ = [re.get() for re in res]
    p.close()
    p.join()


def copy_files(file,root_raw,root_output):
    shutil.copy(
        join(root_raw, file.replace(".nii.gz", ""), "image.nii.gz"),
        join(root_output, "imagesTs", file).replace(".nii.gz", "_0000.nii.gz"),
    )
    shutil.copy(join(root_raw, file.replace(".nii.gz", ""), "seg.nii.gz"),
                join(root_output, "labelsTs", file))

def copy_missing_files(root_raw, root_output,root_changes,files_to_ignore):
    data = pd.read_excel(join(root_changes, "TestsetInspection.xlsx"))
    files=list(data["PATIENT_PSEUDONYM"])
    # files_to_ignore = list(
    #     data[data["Action"].isin(["Remove", "Update_Segmentation", "Update_Label"])]["Name"]
    # )
    #files = os.listdir(root_raw)
    print(files)
    print(files_to_ignore)
    print(f"{len(np.unique(files))} files in total and {len(np.unique(files_to_ignore))} to ignore")
    files = [file for file in files if file not in files_to_ignore]

    print(f"{len(np.unique(files))} Files remain")

    p = Pool(8)
    res=[]
    for file in tqdm(files):
        res.append(
            p.starmap_async(
                copy_files,
                ((file, root_raw,root_output),),
            )
        )
    _ = [re.get() for re in res]
    p.close()
    p.join()

def merge_files(root_output,still_wrong):
    files=os.listdir(join(root_output,"labelsTs_seg_updated"))
    for file in tqdm(files):
        if file.replace("_0000","") in still_wrong:
            continue
        shutil.copy(join(root_output,"labelsTs_seg_updated",file),join(root_output,"labelsTs",file))
        shutil.copy(join(root_output,"imagesTs_seg_updated",file.replace(".nii.gz","_0000.nii.gz")),join(root_output,"imagesTs",file.replace(".nii.gz","_0000.nii.gz")))

    # files = os.listdir(join(root_output, "labelsTr_labels_updated"))
    # for file in tqdm(files):
    #     if file in still_wrong:
    #         continue
    #     shutil.copy(join(root_output, "labelsTr_labels_updated", file),
    #                 join(root_output, "labelsTr", file))
    #     shutil.copy(
    #         join(root_output, "imagesTr_labels_updated", file.replace(".nii.gz", "_0000.nii.gz")),
    #         join(root_output, "imagesTr", file.replace(".nii.gz", "_0000.nii.gz")))

if __name__ == "__main__":
    root_input = "/home/l727r/Documents/E132-Rohdaten/nnUNetv2/Dataset162_ShankTeeth"

    root_output = "/home/l727r/Documents/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/RadboudUni_2022_ShankCBCTTeeth/Dataset163_ShankTeeth_v2"


    root_changes = "/home/l727r/Documents/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/RadboudUni_2022_ShankCBCTTeeth/Dataset_163_TestSet_Changelog"
    root_raw = "/home/l727r/Documents/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/RadboudUni_2022_ShankCBCTTeeth/raw_data"

    # Copy missing information from root_raw to root_changes/output
    # copy_information_from_raw_to_changed(root_raw, join(root_changes, "output", "new_labels"))

    # Generate the new imgs and segs for root_changes/output (in Fabians Format)
    # generate_img_and_seg(join(root_changes,"output","new_labels"))

    # Copy them to the output folder
    # copy_updated_labels(join(root_changes,"output","new_labels"),root_output)

    # Convert segmentations to Fabian_convertion and copy to output
    #convert_seg_to_Fabi(join(root_changes,"gt_corrected"),root_output,root_raw)

    # Copy all missing and not removed images to output folder
    still_wrong=["drusi-interim-toad.nii.gz","elnore-printed-lion.nii.gz","estrellita-federal-gorilla.nii.gz","eustacia-frightened-toucan.nii.gz"]
    memorry_error=[
        "maria-fatal-moose.nii.gz", # 1
        "marion-cold-rhinoceros.nii.gz", # 2
        "shay-secret-parrotfish.nii.gz", # 3
        "torey-square-caterpillar.nii.gz", # 4
        "bernita-dusty-narwhal.nii.gz", # 5
        "bianca-clumsy-orangutan.nii.gz", # 6
        "binni-equivalent-termite.nii.gz", # 7
        "alexa-precious-rhinoceros.nii.gz", # 8
        "cherin-symbolic-dragonfly.nii.gz", # 9
        "delphinia-aggressive-hyena.nii.gz", # 10
        "alleen-bitter-zebra.nii.gz", # 11
        "flory-thundering-heron.nii.gz", # 12
        "anabelle-distinguished-mandrill.nii.gz", # 13
        "jo-ambitious-albatross.nii.gz", # 14
    ]
    updated=os.listdir(join(root_output,"labelsTs_seg_updated"))
    updated=[up.replace("_0000","") for up in updated]
    #copy_missing_files(root_raw, root_output, root_changes,still_wrong+memorry_error+updated)

    #Merg everthink together

    merge_files(root_output,still_wrong+memorry_error)
