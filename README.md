# Prerequisites
Install all dependencies. Have a working nnU-Net setup!

# Prepare the internal dataset and the Toothfairy2 Dataset
Inhouse dataset preparation is done in [this folder](toothseg/datasets/inhouse_dataset). Since it cannot be released 
this code is unlikely to be useful for you.

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
```bash
# Inhouse
# semantic branch
nnUNetv2_train 181 3d_fullres_resample_torch_256_bs8 all -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 4
# instance branch
nnUNetv2_train 188 3d_fullres_resample_torch_192_bs8 all -tr nnUNetTrainer -num_gpus 4

# ToothFairy2
# semantic branch
nnUNetv2_train 121 3d_fullres_resample_torch_256_bs8 all -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 4
# instance branch
nnUNetv2_train 123 3d_fullres_resample_torch_192_bs8 all -tr nnUNetTrainer -num_gpus 4
```

# Testing
# Run test set predictions

1. Resize the test set for the instance segmentation (border-core) prediction via `resize_test_set.py`. This is not needed for semantic seg
2. Run the predictions. Take inspiration from `predict_test_set.sh`
     1. Semantic segmentations will already be in the correct spacing (same as original images)
     2. Border-core instance segmentations are in their 0.2x0.2x0.2 spacing (needed for proper conversion) and must be processed further
3. Convert the border-core prediction into instances with `cbctseg/proposed_method/postprocess_predictions/border_core_to_instances.py`
4. Resize instances to original spacing with `cbctseg/proposed_method/postprocess_predictions/resize_predictions.py`
5. Now we can assign the predicted instances the correct tooth labels (as predicted by the semantic segmentation model) with `cbctseg/proposed_method/postprocess_predictions/assign_tooth_labels.py`

# Evaluate test set

1. Evaluate the quality of the instances (regardless of tooth label) with `cbctseg/evaluation/evaluate_instances.py`
2. Evaluate the quality of the predictions (instances AND correct tooth label) with `cbctseg/evaluation/evaluate_instances_with_tooth_label.py`

Our dataset preparation already did resampling. This is necessary because border/core needs a consistent border size. 
Converting to border core and then letting nnU-Net do the resampling would lead to inconsistent border thicknesses. 
So that's da wae.

Evaluation should (as always) be done in the original image size. So we need to resize all images back to the original 
image size before doing anything else with them. This is what the scripts in this folder are for.

Instructions:

1. Semantic segmentation
   - resize to original sizes with resize_predictions.py
2. Border-Core (instance) segmentations
   - first convert border-core representation to instances with border_core_to_instances.py
   - then convert back to original sizes with resize_predictions.py
3. Bring it all together with assign_tooth_labels.py