from multiprocessing import set_start_method
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.Dataset119_ToothFairy2_All import process_ds, mapping_DS121
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from toothseg.datasets.inhouse_dataset.instance_segmentation_branch_data import convert_sem_dataset_to_instance
from toothseg.datasets.inhouse_dataset.semantic_segmentation_branch_data import convert_dataset

if __name__ == '__main__':
    DOWNLOADED_TOOTHFAIRY2_DIR = ('/media/l727r/data/nnUNet/nnUNetv2_raw/')  # this must be dataset 112
    # Different nnUNet Datasets
    # Dataset 112: Raw
    # Dataset 119: Replace NaN classes
    # Dataset 120: Only Teeth + Jaw Classes
    # Dataset 121: Only Teeth Classes

    # Dataset 121 has only the teeth and disregards the other classes we are not interested in
    process_ds(DOWNLOADED_TOOTHFAIRY2_DIR, "Dataset112_ToothFairy2", "Dataset121_ToothFairy2_Teeth", mapping_DS121(), None)

    #set_start_method('spawn')
    source_dataset = maybe_convert_to_dataset_name(121)
    source_dir = join(nnUNet_raw, source_dataset)

    # this is for the evaluation in our paper. We need to be consistent with our method, so we cannot alter the
    # spacing even though this would make sense for the challenge. Our dev dataset has better spacings than the
    # challenge which is why lower spacings were beneficial for us.
    convert_dataset(
        source_dir,
        f'Dataset122_ToothFairy2fixed_teeth_spacing02',
        (0.2, 0.2, 0.2),
        2,
        6
    )

    convert_sem_dataset_to_instance(
        maybe_convert_to_dataset_name(122),
        'Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px',
        0.2,
        3,
        num_processes=96
    )
    # just here for our convenience
    # export nnUNet_raw='/home/isensee/drives/E132-Projekte/Projects/2024_MICCAI24_ToothFairy2/nnUNet_raw'
    # export nnUNet_raw=/omics/groups/OE0441/E132-Projekte/Projects/2024_MICCAI24_ToothFairy2/nnUNet_raw

