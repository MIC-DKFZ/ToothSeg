from multiprocessing import set_start_method
from typing import Tuple, Iterable
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

from toothseg.evaluation.evaluate_instances import compute_matches_folders


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

def count_classes(matches:List[Tuple],num_classes:int,index=0):
    """
    count how often each class occurs in the given matches
    """
    counts = np.zeros(num_classes,dtype=int)
    for case in matches:
        ids = [c[index]-1 for c in case if c[index] is not None] # due to bg class we start at 1
        counts[ids] += 1
    return counts

def count_class(matches:List[Tuple],label:int,index:int):
    """
    count how often each class occurs in the given matches
    index==0: count GT
    index==1: count Pred
    """
    count = 0
    for m in matches:
        for mi in m:
            if mi[index] == label:
                count += 1
    return count

def compute_obj_metrics(folder_pred, folder_gt, num_processes: int, labels: Iterable[int], file_ending: str = '.nii.gz'):
    # Find Matches between GT and Pred
    matches = compute_matches_folders(folder_pred, folder_gt, num_processes=num_processes, file_ending=file_ending)
    # Count how often a class appears in GT and Prediction
    all_count_gt = {i:count_class(matches,i,0)for i in labels}
    all_count_pred ={i:count_class(matches,i,1)for i in labels}
    # Compute Metric - objF1 and TPDice
    all_obj_f1 = {i: object_level_f1(matches, i) for i in labels}
    all_tp_dice = {i: compute_dice_tp(matches, i) for i in labels}
    # Correct Metric for FN and FP
    for i in labels:
        # TN: (!inGT & !inP) == (TPDice == NaN & F1 == NaN)    --> TP = NaN, F1=NaN     --> PDice=NaN
        # FN: ( inGT & !inP) == (TPDice == NaN & F1 == 0)      --> TP = 0,   F1=0       --> PDice=0
        # TP: ( inGT &  inP) == (TPDice == X   & F1 == Y)      --> TP = X,   F1 = Y     --> PDice=X*Y
        # FP: (!inGT &  inP) == (TPDice == NaN & F1 == 0)      --> TP = NaN, F1=NaN     --> PDice=NaN
        inGT=all_count_gt[i]!=0
        inP=all_count_pred[i]!=0
        if inGT and not inP: # FN - Set TPDice to 0.0
            all_tp_dice[i] = 0.0
        elif not inGT and inP: # FP - Set FP to Nan
            all_obj_f1[i] = np.NaN

    all_pan_dice = {i: all_obj_f1[i] * all_tp_dice[i] for i in labels}

    avg_obj_f1 = np.nanmean(list(all_obj_f1.values()))
    avg_tp_dice = np.nanmean(list(all_tp_dice.values()))
    avg_pan_dice = np.nanmean(list(all_pan_dice.values()))

    metrics = {
        'avg_obj_f1': avg_obj_f1,
        'avg_tp_dice': avg_tp_dice,
        'avg_panoptic_quality_dice': avg_pan_dice,
        'gt_folder': folder_gt,
        'obj_f1_by_label': all_obj_f1,
        'tp_dice_by_label': all_tp_dice,
        'panoptic_quality_dice_by_label': all_pan_dice
    }
    save_json(metrics, join(folder_pred, 'metrics_obj.json'), sort_keys=False)
    print(folder_pred)
    print('avg_obj_f1', metrics['avg_obj_f1'])
    print('avg_tp_dice', metrics['avg_tp_dice'])
    print('avg_panoptic_quality_dice', metrics['avg_panoptic_quality_dice'])


if __name__ == '__main__':
    set_start_method('spawn')

    import argparse
    parser = argparse.ArgumentParser("This script takes a folder containing files with tooth predictions "
                                     "and evaluates them vs the "
                                     "reference. Tooth label matters!"
                                     "Metrics will be saved in metrics_obj.json in input folder\n"
                                     "Requires predicted and reference files to have the same shapes!")
    parser.add_argument('-i', type=str, required=True,
                        help='Input folder. Must contain files with tooth predictions (correct tooth labels!)')
    parser.add_argument('-ref', type=str, required=True,
                        help="Reference folder. Must contain files with the same name as the segmentations in -i.")
    parser.add_argument('-fe', type=str, required=False, default='.nii.gz',
                        help=f'File ending, Default: .nii.gz')
    parser.add_argument('-np', type=int, required=False, default=8,
                        help='Number of processes used for multiprocessing. Default: 8. Make this a lot higher if you '
                             'can!')
    args = parser.parse_args()
    compute_obj_metrics(args.i, args.ref, num_processes=args.np, labels=list(range(1, 33)), file_ending=args.fe)

    # folders_pred = [
    #     # semantic segs
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset164_Filtered_Classes/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch/fold_0/validation_resized',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset164_Filtered_Classes/nnUNetTrainer__nnUNetPlans__3d_lowres_resample_torch/fold_0/validation_resized',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset164_Filtered_Classes/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_resized',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_resized',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_resized',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_resized',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_resized',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs16/fold_0/validation_resized',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8/fold_0/validation_resized',
    #
    #     # instance segs
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances_resized_181_sp03_256_bs8_mintoothsize0_isoFalse_allowbgTrue',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances_resized_181_sp03_256_bs8_mintoothsize0_isoFalse_allowbgTrue',
    #     '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs16/fold_0/validation_instances_resized_181_sp03_256_bs8_mintoothsize0_isoFalse_allowbgTrue',
    #     # '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_256_bs8/fold_0/validation_instances_resized_181_sp03_256_bs8_mintoothsize0_isoFalse_allowbgTrue',
    # ]
    #
    # original_seg_folder_tr = '/dkfz/cluster/gpu/data/OE0441/isensee/Shank_testSet/original_segs/labelsTr'
    # for f in folders_pred:
    #     compute_obj_metrics(f, original_seg_folder_tr, num_processes=190, labels=list(range(1, 33)))
