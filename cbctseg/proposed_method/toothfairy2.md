# Dataset Prep
1. Convert dataset to nifti with `convert_to_nifti.py`
2. Remove NA labels with `remove_na_classes.py`
3. Create teeth only dataset with `teeth_only.py`. This will be our semantic dataset
4. Run `prepare_dataset/toothfairy2.py` to create the instance seg dataset. This requires a GPU and lot of CPUs! Be smart about how you configure this script and where you run it

# nnU-Net planning and preprocessing
- `nnUNetv2_extract_fingerprint -d 116 118 -np 64`
- `nnUNetv2_plan_experiment -d 116 118`

# Manual plans editing
Paste the snippets from [here](readme.md#instance-segmentation-plans) into Dataset118 and 
from [here](readme.md#semantic-segmentation-plans) into Dataset116 configurations!

# Preprocessing
- `nnUNetv2_preprocess -d 118 -c 3d_fullres_resample_torch_192_bs8 -np 32`
- `nnUNetv2_preprocess -d 116 -c 3d_fullres_resample_torch_256_bs8 -np 32`

# Run trainings
Semantic seg:\
`nnUNetv2_train 116 3d_fullres_resample_torch_256_bs8 0 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 8`

bsub -gpu num=8:j_exclusive=yes:gmem=33G -q gpu "source ~/load_env_torch221.sh && nnUNetv2_train 116 3d_fullres_resample_torch_256_bs8 0 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -num_gpus 8"

Instance seg:\
`nnUNetv2_train 118 3d_fullres_resample_torch_192_bs8 0 -num_gpus 4`


bsub -gpu num=4:j_exclusive=yes:gmem=33G -q gpu "source ~/load_env_torch221.sh && nnUNetv2_train 118 3d_fullres_resample_torch_192_bs8 0 -num_gpus 4"

# Training evaluation
1. Collect the cross-validation results so that they are all in one folder:
  - `nnUNetv2_accumulate_crossval_results 118 -c 3d_fullres_resample_torch_192_bs8`
  - `nnUNetv2_accumulate_crossval_results 116 -c 3d_fullres_resample_torch_256_bs8 -tr nnUNetTrainer_onlyMirror01_DASegOrd0`
2. Convert border-core to instances: 
    ```bash
    export nnUNet_raw=/omics/groups/OE0441/E132-Projekte/Projects/2024_MICCAI24_ToothFairy2/nnUNet_raw
    BASE=${nnUNet_results}/Dataset118_ToothFairy2fixed_teeth_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8
    REF=${nnUNet_raw}/Dataset115_ToothFairy2fixed/imagesTr
    python postprocess_predictions/border_core_to_instances.py -i ${BASE}/crossval_results_folds_0_1_2_3_4 -o ${BASE}/crossval_results_folds_0_1_2_3_4_instances -np 64
    python postprocess_predictions/resize_predictions.py -i ${BASE}/crossval_results_folds_0_1_2_3_4_instances -o ${BASE}/crossval_results_folds_0_1_2_3_4_instances_resized -ref ${REF} -np 64
   ```
3. Apply tooth labels
   ```bash
   IFOLDER=${nnUNet_results}/Dataset118_ToothFairy2fixed_teeth_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/crossval_results_folds_0_1_2_3_4_instances_resized
   SFOLDER=${nnUNet_results}/Dataset116_ToothFairy2fixed_teeth/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8/crossval_results_folds_0_1_2_3_4
   OFOLDER=${nnUNet_results}/Dataset118_ToothFairy2fixed_teeth_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/final_predictions_merged
   python postprocess_predictions/assign_tooth_labels.py -ifolder ${IFOLDER} -sfolder ${SFOLDER} -o ${OFOLDER} -np 64
   ```
4. Evaluate\
   Instances only
   ```bash
   python evaluation/evaluate_instances.py -i ${OFOLDER} -ref ${nnUNet_raw}/Dataset116_ToothFairy2fixed_teeth/labelsTr -np 64
   ```
   With Tooth label
   ```bash
   python evaluation/evaluate_instances_with_tooth_label.py -i ${OFOLDER} -ref ${nnUNet_raw}/Dataset116_ToothFairy2fixed_teeth/labelsTr -np 64
   ```
   
   Just the semseg for comparison with our merged labels. Should show a small but consistent gain across metrics for the merged labels
   ```bash
   python evaluation/evaluate_instances.py -i ${SFOLDER} -ref ${nnUNet_raw}/Dataset116_ToothFairy2fixed_teeth/labelsTr -np 64
   python evaluation/evaluate_instances_with_tooth_label.py -i ${SFOLDER} -ref ${nnUNet_raw}/Dataset116_ToothFairy2fixed_teeth/labelsTr -np 64
   ```