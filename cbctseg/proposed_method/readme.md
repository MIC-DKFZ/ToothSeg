This folder contains all the information necessary to reproduce the results of the proposed method

# Prepare the internal dataset and the Toothfairy2 Dataset
TODO

# Fingerprint extraction and preprocessing
nnUNetv2_extract_fingerprint -d 181 188 -np 64
nnUNetv2_plan_experiment -d 181 188

# Add the following configurations to your semantic segmentation plans (Dataset181)

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

# Add the following configurations to your semantic segmentation plans (Dataset181)
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
nnUNetv2_preprocess -d 181 -c 3d_fullres_resample_torch_256_bs8 -np 64
nnUNetv2_preprocess -d 188 -c 3d_fullres_resample_torch_192_bs8 -np 64
```

# Training

## Internal dataset
```bash
# semantic branch
nnUNetv2_train 181 3d_fullres_resample_torch_256_bs8 all -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 4
# instance branch
nnUNetv2_train 188 3d_fullres_resample_torch_192_bs8 all -tr nnUNetTrainer -num_gpus 4
```

