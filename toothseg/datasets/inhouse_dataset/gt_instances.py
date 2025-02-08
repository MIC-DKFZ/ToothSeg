import argparse
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from toothseg.datasets.toothfairy2.gt_instances import determine_fdi_pair_distributions
from toothseg.toothseg.postprocess_predictions.border_core_to_instances import convert_all_sem_to_instance
from toothseg.toothseg.postprocess_predictions.resize_predictions import resample_segmentations_to_ref


if __name__ == '__main__':
    assert Path(join(nnUNet_raw, "Dataset166_CBCTTeeth_instance_spacing02_brd3px")).exists(), (
        'Please set your nnUNet_raw environment variable to the folder containg the in-house '
        'dataset and run the dataset conversions first, see toothseg/datasets/'
        'inhouse_dataset/instance_segmentation_branch_data.py for the conversion script.'
    )

    # convert border-core back to instances
    border_core_seg_folder = join(nnUNet_raw, "Dataset166_CBCTTeeth_instance_spacing02_brd3px", "labelsTr")
    instance_seg_folder = join(nnUNet_raw, "Dataset167_CBCTTeeth_instance_spacing02", "labelsTr")
    convert_all_sem_to_instance(border_core_seg_folder, instance_seg_folder, overwrite=False,
                            small_center_threshold=16,
                            isolated_border_as_separate_instance_threshold=0,
                            num_processes=16, min_instance_size=16,
                            file_ending='.nii.gz')
    
    # convert instances back to original spacing    
    source_dataset = maybe_convert_to_dataset_name(164)
    sem_seg_folder = join(nnUNet_raw, source_dataset, 'labelsTr')
    source_seg_folder = join(nnUNet_raw, "Dataset168_CBCTTeeth_instance", "labelsTr")
    resample_segmentations_to_ref(sem_seg_folder, instance_seg_folder, source_seg_folder, overwrite=False,
                            num_threads_cpu_resampling=32)

    # get training split to determine tooth pair distances for
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=str, choices=['ablation', 'test'],
                        help='Set the analysis to get results for to only use train files.')
    args = parser.parse_args()
    
    if args.analysis == 'ablation':
        fold = 0
        with open('toothseg/datasets/inhouse_dataset/splits_final.json', 'r') as f:
            split = json.load(f)[fold]['train']
        files = [f for f in Path(sem_seg_folder).glob('*.nii.gz') if f.name[:-7] in split] 
    elif args.analysis == 'test':
        files = sorted(Path(sem_seg_folder).glob('*.nii.gz'))
    else:
        raise ValueError('Please specify either "ablation" or "test.')  

    # determine means and covariance matrices of multi-varite Gaussians
    pair_means, pair_covs = determine_fdi_pair_distributions(
        Path(source_seg_folder), files
    )
    
    # save modeled tooth pair distributions to storage
    out = {'means': pair_means.tolist(), 'covs': pair_covs.tolist()}
    with open(f'toothseg/datasets/inhouse_dataset/{args.analysis}_fdi_pair_distrs.json', 'w') as f:
        json.dump(out, f)
