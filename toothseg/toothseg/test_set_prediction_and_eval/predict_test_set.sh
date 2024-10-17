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
