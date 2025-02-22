from multiprocessing import set_start_method
from pathlib import Path
from typing import List, Tuple

from acvl_utils.instance_segmentation.instance_matching import compute_all_matches
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


def load_fn(fname: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(fname)).astype(np.uint8)


def compute_matches_folders(
    folder_pred,
    folder_gt,
    num_processes: int = 6,
    overwrite: bool = True,
    sample: str = '',
):
    if not overwrite and isfile(join(folder_pred, 'matches.pkl')):
        matches = load_pickle(join(folder_pred, 'matches.pkl'))
        return matches
        
    files_pred = [p.name for p in Path(folder_pred).glob('*.nii.gz')]
    files_gt = [join(folder_gt, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]

    if sample:
        with open(sample, 'r') as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines if line.strip()]
        files_gt = [f for f in files_gt if f.split('/')[-1][:-7] in lines]
        files_pred = [f for f in files_pred if f.split('/')[-1][:-7] in lines]
        
    matches = compute_all_matches(files_gt, files_pred, load_fn, 0.1, consume_instances=True, num_processes=num_processes)
    save_pickle(matches, join(folder_pred, 'matches.pkl'))

    return matches


def object_level_f1(matches: List[Tuple]) -> float:
    """
    computes the object-level F1 score as a measure for how reliable the objects have been identified
    (only evaluates the presence/absence of objects and does not take into account what class each object is)

    :param matches:
    :return:
    """
    tp = 0
    fp = 0
    fn = 0
    for m in matches:
        for mi in m:
            if mi[0] is None:
                fp += 1
            elif mi[1] is None:
                fn += 1
            elif mi[0] is not None and mi[1] is not None:
                tp += 1
            else:
                raise RuntimeError('tf are false negatives in instance segmentation? Something is wrong with your '
                                   'matches. Install tinder or sth.')
    # print(tp, fp, fn)
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else np.nan


def compute_instance_dice(matches: List[Tuple], use_fp: bool = True, use_fn: bool = True) -> float:
    """
    compute the average Dice score of instances

    IGNORES TOOTH LABEL, JUST CHECKS INSTANCES

    :param matches:
    :return:
    """
    dc_vals = []
    for m in matches:
        for mi in m:
            if mi[0] is None:
                if use_fp:
                    dc_vals.append(mi[2])
            elif mi[1] is None:
                if use_fn:
                    dc_vals.append(mi[2])
            elif mi[0] is not None and mi[1] is not None:
                dc_vals.append(mi[2])
            else:
                raise RuntimeError('tf are false negatives in instance segmentation? Something is wrong with your '
                                   'matches. Install tinder or sth.')
    return np.mean(dc_vals)


def compute_instance_only_metrics(folder_pred, folder_gt, num_processes: int, sample: str):
    matches = compute_matches_folders(folder_pred, folder_gt, num_processes=num_processes, sample=sample)
    inst_f1 = object_level_f1(matches)
    avg_tp_dice = compute_instance_dice(matches, use_fp=False, use_fn=False)
    avg_dice = compute_instance_dice(matches, use_fp=True, use_fn=True)
    metrics = {
        'gt_folder': folder_gt,
        'inst_f1': inst_f1,
        'avg_tp_dice': avg_tp_dice,
        'avg_dice': avg_dice,
        'panoptic_dice': inst_f1 * avg_tp_dice
    }
    save_json(metrics, join(folder_pred, 'metrics_inst.json'))
    print(folder_pred)
    for k, v in metrics.items():
        print(k, v)
    return metrics


if __name__ == '__main__':
    set_start_method('spawn')

    import argparse
    parser = argparse.ArgumentParser("This script takes a folder containing nifti files with instance predictions "
                                     "(tooth label doesn't matter) and evaluates the quality of the instances vs the "
                                     "reference. Useful for measuring how well we recognize teeth in general."
                                     "Metrics will be saved in metrics_inst.json in input folder.\n"
                                     "Requires predicted and reference files to have the same shapes!")
    parser.add_argument('-i', type=str, required=True,
                        help="Input folder. Must contain nifti files with instance predictions (tooth label doesn't "
                             "matter)")
    parser.add_argument('-ref', type=str, required=True,
                        help="Reference folder. Must contain files with the same name as the segmentations in -i.")
    parser.add_argument('-np', type=int, required=False, default=8,
                        help='Number of processes used for multiprocessing. Default: 8. Make this a lot higher if you '
                             'can!')
    parser.add_argument('-sample', type=str, required=False, default='',
                        help='File with scan names to only evaluate a subsample. Default: ""')
    args = parser.parse_args()

    # original_seg_folder_tr = '/dkfz/cluster/gpu/data/OE0441/isensee/Shank_testSet/original_segs/labelsTr'
    compute_instance_only_metrics(args.i, args.ref, num_processes=args.np, sample=args.sample)