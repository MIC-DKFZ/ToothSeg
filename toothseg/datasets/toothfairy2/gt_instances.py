from collections import defaultdict
from functools import partial
import multiprocessing as mp
from pathlib import Path
from typing import List

from batchgenerators.utilities.file_and_folder_operations import *
import matplotlib.pyplot as plt
import nibabel
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from toothseg.toothseg.postprocess_predictions.border_core_to_instances import convert_all_sem_to_instance
from toothseg.toothseg.postprocess_predictions.resize_predictions import resample_segmentations_to_ref


def process_case(source_seg_folder: Path, sem_file: Path):
    sem_nii = nibabel.load(sem_file)
    sem_seg = np.asarray(sem_nii.dataobj)
    orientation = nibabel.io_orientation(sem_nii.affine)
    assert np.all(orientation[:, 0] == [0, 1, 2])
    spacing = np.array(sem_nii.header.get_zooms())
    sem_seg = nibabel.apply_orientation(sem_seg, orientation)
    shape = sem_seg.shape

    inst_file = source_seg_folder / sem_file.name
    inst_nii = nibabel.load(inst_file)
    inst_seg = np.asarray(inst_nii.dataobj)
    orientation = nibabel.io_orientation(inst_nii.affine)
    inst_seg = nibabel.apply_orientation(inst_seg, orientation)

    # determine voxel counts of instances
    instances = np.unique(inst_seg)[1:]
    voxel_counts = ndimage.sum_labels(np.ones_like(inst_seg), inst_seg, instances)

    # determine FDI number of each instance
    inst_fg_seg = inst_seg.copy()
    inst_fg_seg[sem_seg == 0] = 0
    sem_sums = ndimage.sum_labels(sem_seg, inst_fg_seg, instances)
    inst_fdis = np.round(sem_sums / voxel_counts).astype(int) - 1

    # determine centroid of each instance
    x_sums = ndimage.sum_labels(
        np.tile(np.arange(shape[0])[:, None, None], (1, *shape[1:])),
        inst_seg, instances,
    )
    y_sums = ndimage.sum_labels(
        np.tile(np.arange(shape[1])[None, :, None], (shape[0], 1, shape[2])),
        inst_seg, instances,
    )
    z_sums = ndimage.sum_labels(
        np.tile(np.arange(shape[2])[None, None, :], (*shape[:2], 1)),
        inst_seg, instances,
    )
    xyz_sums = np.stack((x_sums, y_sums, z_sums), axis=1)
    inst_centroids = xyz_sums / voxel_counts[:, None] * spacing

    # determine tooth pair offsets
    pair_offsets = defaultdict(list)
    for i in range(inst_fdis.shape[0]):
        for j in range(i + 1, inst_fdis.shape[0]):
            offsets = inst_centroids[j] - inst_centroids[i]

            # track offsets in both directions
            fdi1, fdi2 = inst_fdis[i], inst_fdis[j]
            pair_offsets[fdi1, fdi2].append(offsets)
            pair_offsets[fdi2, fdi1].append(-offsets)

            # track offsets in both reflections
            fdi1 = fdi1 - 8 if (fdi1 % 16) >= 8 else fdi1 + 8
            fdi2 = fdi2 - 8 if (fdi2 % 16) >= 8 else fdi2 + 8
            offsets = offsets.copy()
            offsets[0] *= -1
            pair_offsets[fdi1, fdi2].append(offsets)
            pair_offsets[fdi2, fdi1].append(-offsets)

    return pair_offsets


def remove_outliers(
    pair_offsets,
    axes: List[int, '...']=[0, 1],
    poly_degree: int=3,
    max_diffs: List[int, int]=[16, 22],  # upper, lower
):
    reg_functions = []
    for is_arch_lower in [False, True]:
        gap2dists = defaultdict(lambda: np.zeros(0))
        for fdi1 in range(16):
            for fdi2 in range(fdi1, 16):
                offsets = pair_offsets[16 * is_arch_lower + fdi1, 16 * is_arch_lower + fdi2]
                dists = np.linalg.norm(offsets[:, axes], axis=-1)
                lr1 = 7 - fdi1 if fdi1 < 8 else fdi1
                lr2 = 7 - fdi2 if fdi2 < 8 else fdi2
                gap = np.abs(lr2 - lr1)
                gap2dists[gap] = np.concatenate((gap2dists[gap], dists))

        xs = np.arange(16).repeat([len(gap2dists[gap]) for gap in range(16)])
        ys = [dist for gap in range(16) for dist in gap2dists[gap]]
        fit = np.polyfit(xs, ys, poly_degree)
        reg_functions.append(np.poly1d(fit))

    clean_pair_offsets = defaultdict(list)
    for (fdi1, fdi2), offsets in pair_offsets.items():
        if fdi1 // 16 != fdi2 // 16:
            continue

        mod_fdi1, mod_fdi2 = fdi1 % 16, fdi2 % 16
        lr1 = 7 - mod_fdi1 if mod_fdi1 < 8 else mod_fdi1
        lr2 = 7 - mod_fdi2 if mod_fdi2 < 8 else mod_fdi2
        gap = np.abs(lr2 - lr1)
        expected_dist = reg_functions[fdi1 // 16](gap)

        dists = np.linalg.norm(offsets[:, axes], axis=-1)
        keep_offsets = [
            direction
            for direction, dist in zip(offsets, dists)
            if np.abs(expected_dist - dist) <= max_diffs[fdi1 >= 16]
        ]
        keep_offsets = np.stack(keep_offsets) if keep_offsets else np.zeros((0, 3))
        clean_pair_offsets[fdi1, fdi2] = keep_offsets

    return clean_pair_offsets


def determine_fdi_pair_distributions(
    source_seg_folder: Path,
    files: List[Path],
    verbose: bool=False,
    num_processes: int=32
):
    # get all tooth pair offsets from training data
    pair_offsets = defaultdict(lambda: np.zeros((0, 3)))
    with mp.Pool(num_processes) as p:
        i = p.imap_unordered(partial(process_case, source_seg_folder), files)
        for offsets in tqdm(i, total=len(files)):
            for k, v in offsets.items():
                pair_offsets[k] = np.concatenate((pair_offsets[k], v))

    # remove offsets between incorrect annotations
    pair_offsets = remove_outliers(pair_offsets)

    # copy same-tooth instances to have samples for each FDI pair
    upper_same_offsets = np.concatenate([pair_offsets[i, i] for i in range(16)])
    for i in range(16):
        pair_offsets[i, i] = upper_same_offsets
    lower_same_offsets = np.concatenate([pair_offsets[i, i] for i in range(16, 32)])
    for i in range(16, 32):
        pair_offsets[i, i] = lower_same_offsets

    # model offsets per FDI pair as multi-variate Gaussian
    pair_means = np.zeros((32, 32, 3))
    pair_covs = np.zeros((32, 32, 3, 3))
    for fdi1 in range(32):
        for fdi2 in range(32):
            if fdi1 // 16 != fdi2 // 16:
                continue

            offsets = pair_offsets[fdi1, fdi2]
            assert len(offsets) > 0            
            pair_means[fdi1, fdi2] = np.mean(offsets, axis=0)
            pair_covs[fdi1, fdi2] = np.cov(offsets.T)

    if not verbose:
        return pair_means, pair_covs

    for i, name in zip(range(3), 'xyz'):
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(name)
        axs[0, 0].imshow(pair_means[:16, :16, i])
        axs[0, 0].set_title('Upper jaw means')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(pair_covs[:16, :16, i, i])
        axs[0, 1].set_title('Upper jaw SDs')
        axs[0, 1].axis('off')
        axs[1, 0].imshow(pair_means[16:, 16:, i])
        axs[1, 0].set_title('Lower jaw means')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(pair_covs[16:, 16:, i, i])
        axs[1, 1].set_title('Lower jaw SDs')
        axs[1, 1].axis('off')
        plt.show(block=True)

    return pair_means, pair_covs


if __name__ == '__main__':
    assert Path(join(nnUNet_raw, "Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px")).exists(), (
        'Please set your nnUNet_raw environment variable to the folder containg the ToothFairy2 '
        'challenge dataset and run the dataset conversions first, see '
        'toothseg/datasets/toothfairy2/toothfairy2.py for the conversion script.'
    )

    # convert border-core back to instances
    border_core_seg_folder = join(nnUNet_raw, "Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px", "labelsTr")
    instance_seg_folder = join(nnUNet_raw, "Dataset124_ToothFairy2fixed_teeth_spacing02_instances", "labelsTr")
    convert_all_sem_to_instance(border_core_seg_folder, instance_seg_folder, overwrite=False,
                            small_center_threshold=20,
                            isolated_border_as_separate_instance_threshold=0,
                            num_processes=64, min_instance_size=20,
                            file_ending='.nii.gz')
    
    # convert instances back to original spacing
    source_dataset = maybe_convert_to_dataset_name(121)
    sem_seg_folder = join(nnUNet_raw, source_dataset, 'labelsTr')
    source_seg_folder = join(nnUNet_raw, "Dataset125_ToothFairy2fixed_teeth_instances", "labelsTr")
    resample_segmentations_to_ref(sem_seg_folder, instance_seg_folder, source_seg_folder, overwrite=False,
                            num_threads_cpu_resampling=64)

    # get training files to determine tooth pair distributions for
    fold = 5
    with open('toothseg/datasets/toothfairy2/splits_final.json', 'r') as f:
        split = json.load(f)[fold]['train']
    files = [f for f in Path(sem_seg_folder).glob('*.nii.gz') if f.name[:-7] in split]

    # determine means and covariance matrices of multi-varite Gaussians
    pair_means, pair_covs = determine_fdi_pair_distributions(
        Path(source_seg_folder), files
    )
    
    # save modeled tooth pair distributions to storage
    out = {'means': pair_means.tolist(), 'covs': pair_covs.tolist()}
    with open('toothseg/datasets/toothfairy2/fdi_pair_distrs.json', 'w') as f:
        json.dump(out, f)
