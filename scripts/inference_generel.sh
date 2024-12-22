#!/bin/bash

export nnUNet_compile=F

### 0. Adapt to your needs ###
CODE_PATH=/home/isensee/git_repos/radbouduni_2023_cbctteethsegmentation
input_dir=${nnUNet_raw}/Dataset164_Filtered_Classes   # data needs to be in a imagesTs folder in this directory
output_dir=${nnUNet_results}/CBCT_teeth/testset

### 1. Resize Instance Data ###
# resize the test set to 0.2x0.2x0.2 for the instance segmentation branch. We do not need to resize for the semseg branch
python ${CODE_PATH}/toothseg/toothseg/test_set_prediction_and_eval/resize_test_set.py \
-i ${input_dir}/imagesTs \
-o ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02

### 2.1 semantic segmentation branch ###
# Default
nnUNetv2_predict --save_probabilities -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all
# or Parallelized
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 0 &
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 1 &
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 2 &
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 3 &
#CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 4 &
#CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 5 &
#CUDA_VISIBLE_DEVICES=6 nnUNetv2_predict -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 6 &
#CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i ${input_dir}/imagesTs -o ${output_dir}/semseg_branch -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 7 &
#wait

### 2.2 instances segmentation branch ###
# Default
nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all
# or Parallelized
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 0 &
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 1 &
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 2 &
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 3 &
#CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 4 &
#CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 5 &
#CUDA_VISIBLE_DEVICES=6 nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 6 &
#CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i ${input_dir}/imagesTs_resized_for_instanceseg_spacing_02_02_02 -o ${output_dir}/instseg_branch_border_core -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 7 &
#wait

### 3. Convert border-core to instances ###
NUM_PROCESSES=96
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/border_core_to_instances.py \
-i ${output_dir}/instseg_branch_border_core \
-o ${output_dir}/instseg_branch_border_core_converted_to_instances -np ${NUM_PROCESSES}

### 4. Resize instances to original image spacing ###
NUM_PROCESSES=96
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/resize_predictions.py \
-i ${output_dir}/instseg_branch_border_core_converted_to_instances \
-o ${output_dir}/instseg_branch_border_core_converted_to_instances_resized \
-ref ${input_dir}/imagesTs -np ${NUM_PROCESSES}

### 5. Assign tooth labels ###
NUM_PROCESSES=4
python ${CODE_PATH}/toothseg/toothseg/postprocess_predictions/assign_mincost_tooth_labels.py \
-ifolder ${output_dir}/instseg_branch_border_core_converted_to_instances_resized \
-sfolder ${output_dir}/semseg_branch \
-o ${output_dir}/final_prediction -np ${NUM_PROCESSES}
