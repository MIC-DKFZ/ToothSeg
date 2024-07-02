This folder contains all the information necessary to reproduce the results of the proposed method

1) run the scripts located in [prepare_datasets](prepare_dataset)
2) then run nnU-Net planning and preprocessing
3) then train
4) then


nnUNetv2_extract_fingerprint -d 164 181 182 183 184 185 186 187 188 -np 64
nnUNetv2_plan_experiment -d 164 181 182 183 184 185 186 187 188
nnUNetv2_preprocess -d 164 -c 3d_fullres_resample_torch 3d_lowres_resample_torch -np 64
nnUNetv2_preprocess -d 181 182 183 184 185 186 187 188 -c 3d_fullres_resample_torch_128 -np 64


```bash
conda deactivate
source ~/load_env_torch210.sh
CUDA_VISIBLE_DEVICES=0,1,2,5 nnUNet_def_n_proc=4 nnUNetv2_train 181 3d_fullres_resample_torch_192_bs8 0 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 4
```

# Semantic Segmentation plans

```json
,
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

# Instance Segmentation Plans
```json

,
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