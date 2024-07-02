from multiprocessing import set_start_method
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from cbctseg.proposed_method.prepare_dataset.instance_segmentation import convert_sem_dataset_to_instance
from cbctseg.proposed_method.prepare_dataset.semantic_segmentation import convert_dataset

if __name__ == '__main__':
    # export nnUNet_raw='/home/isensee/drives/E132-Projekte/Projects/2024_MICCAI24_ToothFairy2/nnUNet_raw'
    # export nnUNet_raw=/omics/groups/OE0441/E132-Projekte/Projects/2024_MICCAI24_ToothFairy2/nnUNet_raw
    set_start_method('spawn')
    source_dataset = maybe_convert_to_dataset_name(116)
    source_dir = join(nnUNet_raw, source_dataset)

    convert_dataset(source_dir, f'Dataset117_ToothFairy2fixed_teeth_spacing02', (0.2, 0.2, 0.2))

    convert_sem_dataset_to_instance(
        maybe_convert_to_dataset_name(117),
        'Dataset118_ToothFairy2fixed_teeth_spacing02_brd3px',
        0.2,
        3,
        num_processes=24
    )
