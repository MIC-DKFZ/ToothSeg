source ~/load_env_torch210.sh
export nnUNet_compile=F
# instances as border core with all 5 folds
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -d 188 -c 3d_fullres_resample_torch_192_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 0 &
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -d 188 -c 3d_fullres_resample_torch_192_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 1 &
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -d 188 -c 3d_fullres_resample_torch_192_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 2 &
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -d 188 -c 3d_fullres_resample_torch_192_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 3 &
#CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -d 188 -c 3d_fullres_resample_torch_192_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 4 &
#CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -d 188 -c 3d_fullres_resample_torch_192_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 5 &
#CUDA_VISIBLE_DEVICES=6 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -d 188 -c 3d_fullres_resample_torch_192_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 6 &
#CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -d 188 -c 3d_fullres_resample_torch_192_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 7 &
#wait
#
## instances as border core with single model
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 0 &
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 1 &
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 2 &
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 3 &
#CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 4 &
#CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 5 &
#CUDA_VISIBLE_DEVICES=6 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 6 &
#CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -d 188 -c 3d_fullres_resample_torch_192_bs8 -f all -num_parts 8 -part_id 7 &
#wait

# semantic segmentation of teeth with all 5 folds
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 0 &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 1 &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 2 &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 3 &
CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 4 &
CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 5 &
CUDA_VISIBLE_DEVICES=6 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 6 &
CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f 0 1 2 3 4 -num_parts 8 -part_id 7 &
wait

# semantic segmentation of teeth with single model
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 0 &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 1 &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 2 &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 3 &
CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 4 &
CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 5 &
CUDA_VISIBLE_DEVICES=6 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 6 &
CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -o ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -d 181 -tr nnUNetTrainer_onlyMirror01_DASegOrd0 -c 3d_fullres_resample_torch_256_bs8 -f all -num_parts 8 -part_id 7 &
wait
