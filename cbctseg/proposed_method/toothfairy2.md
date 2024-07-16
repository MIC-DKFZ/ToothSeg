# Dataset Prep
1. Convert dataset to nifti with `convert_to_nifti.py`
2. Remove NA labels with `remove_na_classes.py`
3. Create teeth only dataset with `teeth_only.py`. This will be our semantic dataset
4. Run `prepare_dataset/toothfairy2.py` to create the instance seg dataset. This requires a GPU and lot of CPUs! Be smart about how you configure this script and where you run it

# nnU-Net planning and preprocessing
- `nnUNetv2_extract_fingerprint -d 121 123 -np 64`
- `nnUNetv2_plan_experiment -d 121 123`

# Manual plans editing
Paste the snippets from [here](readme.md#instance-segmentation-plans) into Dataset123 and 
from [here](readme.md#semantic-segmentation-plans) into Dataset121 configurations!

# Preprocessing
- `nnUNetv2_preprocess -d 123 -c 3d_fullres_resample_torch_192_bs8 -np 32`
- `nnUNetv2_preprocess -d 121 -c 3d_fullres_resample_torch_256_bs8 -np 32`

# Run trainings
Semantic seg:\
`nnUNetv2_train 121 3d_fullres_resample_torch_256_bs8 0 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 8`

bsub -gpu num=8:j_exclusive=yes:gmem=33G -q gpu "source ~/load_env_torch221.sh && nnUNetv2_train 121 3d_fullres_resample_torch_256_bs8 0 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 8"

Instance seg:\
`nnUNetv2_train 123 3d_fullres_resample_torch_192_bs8 0 -num_gpus 4`


bsub -gpu num=4:j_exclusive=yes:gmem=33G -q gpu "source ~/load_env_torch221.sh && nnUNetv2_train 123 3d_fullres_resample_torch_192_bs8 0 -num_gpus 4"

# Training evaluation
```bash
export nnUNet_raw=/omics/groups/OE0441/E132-Projekte/Projects/2024_MICCAI24_ToothFairy2/nnUNet_raw
CODE_PATH=/home/isensee/git_repos/radbouduni_2023_cbctteethsegmentation
SEMSEG_DATASET_NAME='Dataset121_ToothFairy2_Teeth'
INSTSEG_DATASET_NAME='Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px'
REF_FOLDER_RESAMPLING=${nnUNet_raw}/Dataset119_ToothFairy2_All/imagesTr

INSTSEG_TRAINER=nnUNetTrainer
SEMSEG_TRAINER=nnUNetTrainer_onlyMirror01_DASegOrd0
INSTSEG_CONFIG=3d_fullres_resample_torch_192_bs2
SEMSEG_CONFIG=3d_fullres_resample_torch_192_bs2
NNUNET_TRAINING_INSTSEG=${INSTSEG_TRAINER}__nnUNetPlans__${INSTSEG_CONFIG}
NNUNET_TRAINING_SEMSEG=${SEMSEG_TRAINER}__nnUNetPlans__${SEMSEG_CONFIG}

nnUNet_def_n_proc=64 nnUNetv2_accumulate_crossval_results ${INSTSEG_DATASET_NAME} -c ${INSTSEG_CONFIG} -tr ${INSTSEG_TRAINER}
nnUNet_def_n_proc=64 nnUNetv2_accumulate_crossval_results ${SEMSEG_DATASET_NAME} -c ${SEMSEG_CONFIG} -tr ${SEMSEG_TRAINER}

# border core to instance
python ${CODE_PATH}/cbctseg/proposed_method/postprocess_predictions/border_core_to_instances.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/crossval_results_folds_0_1_2_3_4 \
-o ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/crossval_results_folds_0_1_2_3_4_instances \
-np 64
# resize border core to ref
python ${CODE_PATH}/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/crossval_results_folds_0_1_2_3_4_instances \
-o ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/crossval_results_folds_0_1_2_3_4_instances_resized \
-ref ${REF_FOLDER_RESAMPLING} -np 64

# apply tooth labels
python ${CODE_PATH}/cbctseg/proposed_method/postprocess_predictions/assign_tooth_labels.py \
-ifolder ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/crossval_results_folds_0_1_2_3_4_instances_resized \
-sfolder ${nnUNet_results}/${SEMSEG_DATASET_NAME}/${NNUNET_TRAINING_SEMSEG}/crossval_results_folds_0_1_2_3_4 \
-o ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/final_predictions_merged \
-np 64

# evaluate
# full eval with tooth labels
python ${CODE_PATH}/cbctseg/evaluation/evaluate_instances_with_tooth_label.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/final_predictions_merged \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64
# instances only
python ${CODE_PATH}/cbctseg/evaluation/evaluate_instances.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/final_predictions_merged \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64

# just the semseg for comparison
python ${CODE_PATH}/cbctseg/evaluation/evaluate_instances_with_tooth_label.py \
-i ${nnUNet_results}/${SEMSEG_DATASET_NAME}/${NNUNET_TRAINING_SEMSEG}/crossval_results_folds_0_1_2_3_4 \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64
# instances only
python ${CODE_PATH}/cbctseg/evaluation/evaluate_instances.py \
-i ${nnUNet_results}/${SEMSEG_DATASET_NAME}/${NNUNET_TRAINING_SEMSEG}/crossval_results_folds_0_1_2_3_4 \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64
```
