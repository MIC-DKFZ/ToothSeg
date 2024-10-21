#!/bin/bash

# Adapt to your needs
CODE_PATH=/home/isensee/git_repos/radbouduni_2023_cbctteethsegmentation
SEMSEG_DATASET_NAME='Dataset121_ToothFairy2_Teeth'
INSTSEG_DATASET_NAME='Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px'
REF_FOLDER_RESAMPLING=${SEMSEG_DATASET_NAME}  # We can use semantic dataset since they have the same spacing

INSTSEG_TRAINER=nnUNetTrainer
SEMSEG_TRAINER=nnUNetTrainer_onlyMirror01_DASegOrd0
INSTSEG_CONFIG=3d_fullres_resample_torch_192_bs8_ctnorm
SEMSEG_CONFIG=3d_fullres_resample_torch_256_bs8_ctnorm
NNUNET_TRAINING_INSTSEG=${INSTSEG_TRAINER}__nnUNetPlans__${INSTSEG_CONFIG}
NNUNET_TRAINING_SEMSEG=${SEMSEG_TRAINER}__nnUNetPlans__${SEMSEG_CONFIG}

python ${CODE_PATH}/toothseg/evaluation/evaluate_instances_with_tooth_label.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_final_predictions_merged \
-ref ${nnUNet_raw}/${REF_FOLDER_RESAMPLING}/labelsTr -np 64
# instances only. This gives TD Dice, F1 and Panoptic Dice, but only on instance level disregarding the correct FDI assignment.
python ${CODE_PATH}/toothseg/evaluation/evaluate_instances.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_final_predictions_merged \
-ref ${nnUNet_raw}/${REF_FOLDER_RESAMPLING}/labelsTr -np 64

# This is an example for how to evaluate just the segmentation branch
# full eval with FDI tooth labels
python ${CODE_PATH}/toothseg/evaluation/evaluate_instances_with_tooth_label.py \
-i ${nnUNet_results}/${SEMSEG_DATASET_NAME}/${NNUNET_TRAINING_SEMSEG}/fold_5/validation \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64
# instances only
python ${CODE_PATH}/toothseg/evaluation/evaluate_instances.py \
-i ${nnUNet_results}/${SEMSEG_DATASET_NAME}/${NNUNET_TRAINING_SEMSEG}/fold_5/validation \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64