# Prerequisites
Install all dependencies. Have a working nnU-Net setup!

# Prepare the internal dataset and the Toothfairy2 Dataset
Inhouse dataset preparation is done in [this folder](toothseg/datasets/inhouse_dataset). Since it cannot be released 
this code is unlikely to be useful for you.
Inhouse dataset variants that are created are the following:
- Dataset164_Filtered_Classes: Our inhouse dataset with just the tooth labels, each image is in its original spacing
- Dataset181_CBCTTeeth_semantic_spacing03: Dataset164 resized so that all images have spacing 0.3x0.3x0.3. Used for 
semantic segmentation branch
- Dataset188_CBCTTeeth_instance_spacing02_brd3px: Dataset164 resized to spacing 0.2x0.2x0.2 and then converted to 
border-core for the instance segmentation branch

ToothFairy2 dataset must be downloaded from this [website](https://ditto.ing.unimore.it/toothfairy2/) and then 
processed with [toothfairy2.py](toothseg/datasets/toothfairy2/toothfairy2.py). You need to adapt the script to your 
needs! The script will create three datasets: 
- Dataset121_ToothFairy2_Teeth: Original ToothFairy Dataset but with only teeth remaining (other structures removed). 
ToothFairy already has all images in spacing 0.3. This is used for the semantic segmentation branch
- Dataset122_ToothFairy2fixed_teeth_spacing02: just an intermediate dataset at spacing 0.2
- Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px: border core representation at spacing 0.2 for the instance branch

# Fingerprint extraction and preprocessing

```bash
# Inhouse Dataset
nnUNetv2_extract_fingerprint -d 181 188 -np 64
nnUNetv2_plan_experiment -d 181 188

# ToothFairy2
nnUNetv2_extract_fingerprint -d 121 123 -np 64
nnUNetv2_plan_experiment -d 121 123
```

# Add the following configurations to your semantic segmentation branch plans (Dataset 181 and 121)

```json
        "3d_fullres_resample_torch_256_bs8": {
            "inherits_from": "3d_fullres",
            "data_identifier": "nnUNetPlans_3d_fullres_resample_torch",
            "resampling_fn_data": "resample_torch_fornnunet",
            "resampling_fn_seg": "resample_torch_fornnunet",
            "resampling_fn_data_kwargs": {
                "is_seg": false,
                "force_separate_z": false,
                "memefficient_seg_resampling": false
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": true,
                "force_separate_z": false,
                "memefficient_seg_resampling": false
            },
            "resampling_fn_probabilities": "resample_torch_fornnunet",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": false,
                "force_separate_z": false,
                "memefficient_seg_resampling": false
            },
            "batch_size": 8,
            "patch_size": [
                256,
                256,
                256
            ],
            "spacing": [
                0.3,
                0.3,
                0.3
            ],
            "n_conv_per_stage_encoder": [
                2,
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "n_conv_per_stage_decoder": [
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "num_pool_per_axis": [
                6,
                6,
                6
            ],
            "pool_op_kernel_sizes": [
                [
                    1,
                    1,
                    1
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ]
            ],
            "conv_kernel_sizes": [
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ]
        ]
        }
```

# Add the following configurations to your instance branch plans (Dataset 123 and 188)
```json
        "3d_fullres_resample_torch_192_bs8": {
            "inherits_from": "3d_fullres",
            "data_identifier": "nnUNetPlans_3d_fullres_resample_torch",
            "resampling_fn_data": "resample_torch_fornnunet",
            "resampling_fn_seg": "resample_torch_fornnunet",
            "resampling_fn_data_kwargs": {
                "is_seg": false,
                "force_separate_z": false,
                "memefficient_seg_resampling": false
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": true,
                "force_separate_z": false,
                "memefficient_seg_resampling": false
            },
            "resampling_fn_probabilities": "resample_torch_fornnunet",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": false,
                "force_separate_z": false,
                "memefficient_seg_resampling": false
            },
            "batch_size": 8,
            "patch_size": [
                192,
                192,
                192
            ],
            "spacing": [
                0.2,
                0.2,
                0.2
            ],
            "n_conv_per_stage_encoder": [
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "n_conv_per_stage_decoder": [
                2,
                2,
                2,
                2,
                2
            ],
            "num_pool_per_axis": [
                5,
                5,
                5
            ],
            "pool_op_kernel_sizes": [
                [
                    1,
                    1,
                    1
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ]
            ],
            "conv_kernel_sizes": [
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ]
            ]
        }
```

# Preprocessing
```bash
# inhouse
nnUNetv2_preprocess -d 181 -c 3d_fullres_resample_torch_256_bs8 -np 64
nnUNetv2_preprocess -d 188 -c 3d_fullres_resample_torch_192_bs8 -np 64
# toothfairy2
nnUNetv2_preprocess -d 121 -c 3d_fullres_resample_torch_256_bs8 -np 64
nnUNetv2_preprocess -d 123 -c 3d_fullres_resample_torch_192_bs8 -np 64
```

# Training
## Inhouse dataset
The inhouse dataset is split into train and test via the imagesTr and imagesTs folders. We train on all training cases 
```bash
# Inhouse
# semantic branch
nnUNetv2_train 181 3d_fullres_resample_torch_256_bs8 all -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 8
# instance branch
nnUNetv2_train 188 3d_fullres_resample_torch_192_bs8 all -tr nnUNetTrainer -num_gpus 4
```

## ToothFairy2
For ToothFairy we didn't split the files into imagesTr and imagesTs because that would have interfered with other 
experiments on that datasets. Instead, we extend the nnU-Net splits_final.json  file with a fifth fold that represents 
the 70:30 split used in our paper. Just copy our [splits](toothseg/datasets/toothfairy2/splits_final.json) into the 
nnUNet_preprocessed directory of Datasets 121 and 123.

```bash
# ToothFairy2
# semantic branch
nnUNetv2_train 121 3d_fullres_resample_torch_256_bs8 5 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 4
# instance branch
nnUNetv2_train 123 3d_fullres_resample_torch_192_bs8 5 -tr nnUNetTrainer -num_gpus 4
```

Note how we train fold 5 here, not all.

# Inference
## Inhouse dataset or other unseen datasets
This how inference needs to be performed:
1. Resize the test set for the instance segmentation (border-core) prediction via 
[resize_test_set.py](toothseg/toothseg/test_set_prediction_and_eval/resize_test_set.py). This is not needed for semantic seg
2. Run the predictions as you normally would with nnU-Net
     1. Semantic segmentations will already be in the correct spacing (same as original images)
     2. Border-core instance segmentations are in their 0.2x0.2x0.2 spacing (needed for proper conversion) and must be processed further
3. Convert the border-core prediction into instances with [border_core_to_instances.py](toothseg/toothseg/postprocess_predictions/border_core_to_instances.py)
4. Resize instances to original spacing with [resize_predictions.py](toothseg/toothseg/postprocess_predictions/resize_predictions.py)
5. Now we can assign the predicted instances the correct tooth labels (as predicted by the semantic segmentation model) with [assign_tooth_labels.py](toothseg/toothseg/postprocess_predictions/assign_tooth_labels.py)

Here is the script with which we did this:

```bash
export nnUNet_compile=F
CODE_PATH=/home/isensee/git_repos/radbouduni_2023_cbctteethsegmentation

# resize the test set to 0.2x0.2x0.2 for the instance segmentation branch. We do not need to resize for the semseg branch
# this script only works out of the box for our inhouse dataset. Adapt it (see `__name__ == '__main__'`) for other datasets
python ${CODE_PATH}/toothseg/toothseg/test_set_prediction_and_eval/resize_test_set.py

# instances segmentation branch
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 0 &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 1 &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 2 &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 3 &
CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 4 &
CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 5 &
CUDA_VISIBLE_DEVICES=6 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 6 &
CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 7 &
wait

# semantic segmentation branch
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 0 &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 1 &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 2 &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 3 &
CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 4 &
CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 5 &
CUDA_VISIBLE_DEVICES=6 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 6 &
CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 7 &
wait

# Merging the instance and semantic segmentation branches
NUM_PROCESSES = 96

# convert border-core to instances
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/border_core_to_instances.py \
-i ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core \
-o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core_converted_to_instances -np ${NUM_PROCESSES}

# resize instances to original image spacing
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/resize_predictions.py \
-i ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core_converted_to_instances \
-o ${nnUNet_results}/CBCT_teeth/testset/instseg_branch_border_core_converted_to_instances_resized \
-ref ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -np ${NUM_PROCESSES}

# assign tooth labels
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/assign_tooth_labels.py \
-ifolder ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model_instances_resized \
-sfolder ${nnUNet_results}/CBCT_teeth/testset/semseg_branch \
-o ${nnUNet_results}/CBCT_teeth/testset/final_prediction -np ${NUM_PROCESSES}

```

## ToothFairy2
Since we coded the train:test split into fold 5 we can just use the predictions generated by nnU-Net in the final validation. 
We don't need to run test set inference.
Don't worry about the fact that nnU-Net interprets our test set as validation set: nnU-Net in the setting used here does 
not use the validation set for anything (no epoch selection etc) that would impact test set integrity.

This is how we postprocessed the Toothfairy predictions. The procedure is the same as for the inhouse dataset.
```bash
export nnUNet_raw=/omics/groups/OE0441/E132-Projekte/Projects/2024_MICCAI24_ToothFairy2/nnUNet_raw
CODE_PATH=/home/isensee/git_repos/radbouduni_2023_cbctteethsegmentation
SEMSEG_DATASET_NAME='Dataset121_ToothFairy2_Teeth'
INSTSEG_DATASET_NAME='Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px'
REF_FOLDER_RESAMPLING=${nnUNet_raw}/Dataset119_ToothFairy2_All/imagesTr

INSTSEG_TRAINER=nnUNetTrainer
SEMSEG_TRAINER=nnUNetTrainer_onlyMirror01_DASegOrd0
INSTSEG_CONFIG=3d_fullres_resample_torch_192_bs8_ctnorm
SEMSEG_CONFIG=3d_fullres_resample_torch_256_bs8_ctnorm
NNUNET_TRAINING_INSTSEG=${INSTSEG_TRAINER}__nnUNetPlans__${INSTSEG_CONFIG}
NNUNET_TRAINING_SEMSEG=${SEMSEG_TRAINER}__nnUNetPlans__${SEMSEG_CONFIG}

# border core to instance
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/border_core_to_instances.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation \
-o ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_instances \
-np 64
# resize border core to ref
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/resize_predictions.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_instances \
-o ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_instances_resized \
-ref ${REF_FOLDER_RESAMPLING} -np 64

# apply tooth labels
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/assign_tooth_labels.py \
-ifolder ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_instances_resized \
-sfolder ${nnUNet_results}/${SEMSEG_DATASET_NAME}/${NNUNET_TRAINING_SEMSEG}/fold_5/validation \
-o ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_final_predictions_merged \
-np 64
```

# Evaluate test set

1. Evaluate the quality of the instances (regardless of tooth label) with [evaluate_instances.py](toothseg/evaluation/evaluate_instances.py)
2. Evaluate the quality of the predictions (instances AND correct tooth label) with [evaluate_instances_with_tooth_label.py](toothseg/evaluation/evaluate_instances_with_tooth_label.py)

Example (following the merging of segmentation and instance branch in ToothFairy2 above)
```bash
# evaluate
# full eval with FDI tooth labels. This gives TD Dice, F1 and Panoptic Dice
python ${CODE_PATH}/toothseg/evaluation/evaluate_instances_with_tooth_label.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_final_predictions_merged \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64
# instances only. This gives TD Dice, F1 and Panoptic Dice, but only on instance level disregarding the correct FDI assignment.
python ${CODE_PATH}/toothseg/evaluation/evaluate_instances.py \
-i ${nnUNet_results}/${INSTSEG_DATASET_NAME}/${NNUNET_TRAINING_INSTSEG}/fold_5/validation_final_predictions_merged \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64

# This is an example for how to evaluate just the segmentation branch
# full eval with FDI tooth labels
python ${CODE_PATH}/toothseg/evaluation/evaluate_instances_with_tooth_label.py \
-i ${nnUNet_results}/${SEMSEG_DATASET_NAME}/${NNUNET_TRAINING_SEMSEG}/fold_5/validation \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64
# instances only
python ${CODE_PATH}/toothseg/evaluation/evaluate_instances.py \
-i ${nnUNet_results}/${SEMSEG_DATASET_NAME}/${NNUNET_TRAINING_SEMSEG}/fold_5/validation \
-ref ${nnUNet_raw}/${SEMSEG_DATASET_NAME}/labelsTr -np 64
```
