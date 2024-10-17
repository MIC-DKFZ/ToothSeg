from multiprocessing import set_start_method
from typing import List, Tuple, Iterable
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

from evaluate import compute_matches_folders


def object_level_f1(matches: List[Tuple], label: int) -> float:
    """

    :param matches:
    :return:
    """
    tp = 0
    fp = 0
    fn = 0
    for m in matches:
        # (instance_id_gt, instance_id_pred, dice, num_pixels_gt_instance, num_pixels_pred_instance)
        for mi in m:
            if mi[0] == label and mi[1] == label:
                tp += 1
            elif mi[0] == label and mi[1] != label:
                fn += 1
            elif mi[0] != label and mi[1] == label:
                fp += 1
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else np.nan


def compute_dice_tp(matches: List[Tuple], label: int) -> float:
    """
    compute the average Dice score of instances

    IGNORES TOOTH LABEL, JUST CHECKS INSTANCES

    :param matches:
    :return:
    """
    dc_vals = []
    for m in matches:
        for mi in m:
            if mi[0] == label and mi[1] == label:
                dc_vals.append(mi[2])
    return np.mean(dc_vals)


def compute_obj_metrics(folder_pred, folder_gt, num_processes: int, sample: str,labels: Iterable[int]):
    matches = compute_matches_folders(folder_pred, folder_gt, num_processes=num_processes, sample=sample)
    all_obj_f1 = {i: object_level_f1(matches, i) for i in labels}
    all_tp_dice = {i: compute_dice_tp(matches, i) for i in labels}
    avg_obj_f1 = np.nanmean(list(all_obj_f1.values()))
    avg_tp_dice = np.nanmean(list(all_tp_dice.values()))
    metrics = {
        'avg_obj_f1': avg_obj_f1,
        'avg_tp_dice': avg_tp_dice,
        'avg_panoptic_quality_dice': avg_obj_f1 * avg_tp_dice,
        'gt_folder': folder_gt,
        'obj_f1_by_label': all_obj_f1,
        'tp_dice_by_label': all_tp_dice,
        'panoptic_quality_dice_by_label': {i: all_obj_f1[i] * all_tp_dice[i] for i in labels}
    }
    save_json(metrics, join(folder_pred, 'metrics_obj.json'), sort_keys=False)
    print(folder_pred)
    print('avg_obj_f1', metrics['avg_obj_f1'])
    print('avg_tp_dice', metrics['avg_tp_dice'])
    print('avg_panoptic_quality_dice', metrics['avg_panoptic_quality_dice'])


if __name__ == '__main__':
    set_start_method('spawn')

    import argparse
    parser = argparse.ArgumentParser("This script takes a folder containing nifti files with tooth predictions "
                                     "and evaluates them vs the "
                                     "reference. Tooth label matters!"
                                     "Metrics will be saved in metrics_obj.json in input folder\n"
                                     "Requires predicted and reference files to have the same shapes!")
    parser.add_argument('-i', type=str, required=True,
                        help='Input folder. Must contain nifti files with tooth predictions (correct tooth labels!)')
    parser.add_argument('-ref', type=str, required=True,
                        help="Reference folder. Must contain files with the same name as the segmentations in -i.")
    parser.add_argument('-np', type=int, required=False, default=8,
                        help='Number of processes used for multiprocessing. Default: 8. Make this a lot higher if you '
                             'can!')
    parser.add_argument('-sample', type=str, required=False, default='',
                        help='File with scan names to only evaluate a subsample. Default: ""')
    args = parser.parse_args()
    compute_obj_metrics(args.i, args.ref, num_processes=args.np, sample=args.sample, labels=list(range(1, 33)))
