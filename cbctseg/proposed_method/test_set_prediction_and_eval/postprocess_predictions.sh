# convert border-core to instances
python cbctseg/proposed_method/postprocess_predictions/border_core_to_instances.py -i ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model_instances -np 64
python cbctseg/proposed_method/postprocess_predictions/border_core_to_instances.py -i ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234 -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234_instances -np 64

# resize instances
python cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model_instances -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model_instances_resized -ref ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -np 128
python cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234_instances -o ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234_instances_resized -ref ${nnUNet_raw}/Dataset164_Filtered_Classes/imagesTs -np 128

# assign tooth labels
python cbctseg/proposed_method/postprocess_predictions/assign_tooth_labels.py -ifolder ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_single_model_instances_resized -sfolder ${nnUNet_results}/CBCT_teeth/testset/semseg_single_model -o ${nnUNet_results}/CBCT_teeth/testset/final_single_model -np 64
python cbctseg/proposed_method/postprocess_predictions/assign_tooth_labels.py -ifolder ${nnUNet_results}/CBCT_teeth/testset/instances_as_border_core_ensemble_fold01234_instances_resized -sfolder ${nnUNet_results}/CBCT_teeth/testset/semseg_ensemble_fold01234 -o ${nnUNet_results}/CBCT_teeth/testset/final_ensemble -np 64
