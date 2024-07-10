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

    # this is for our participation in the challenge. We know that the test set will be spacing 0.4 (this is official
    # information) so there is no point in using higher spacings.
    convert_dataset(source_dir,
                    f'Dataset124_ToothFairy2fixed_teeth_spacing04',
                    (0.4, 0.4, 0.4),
                    4,
                    4)

    # given the higher spacing we don't know what border thickness is going to be good. Needs to be retuned.
    convert_sem_dataset_to_instance(
        maybe_convert_to_dataset_name(124),
        'Dataset125_ToothFairy2fixed_teeth_spacing04_brd2px',
        0.4,
        2,
        num_processes=96
    )

    convert_sem_dataset_to_instance(
        maybe_convert_to_dataset_name(124),
        'Dataset126_ToothFairy2fixed_teeth_spacing04_brd3px',
        0.4,
        3,
        num_processes=96
    )

