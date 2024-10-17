python cbctseg/evaluation/evaluate_instances.py -i ${nnUNet_results}/CBCT_teeth/testset/final_single_model -ref ${nnUNet_raw}/Dataset164_Filtered_Classes/labelsTs -np 64
python cbctseg/evaluation/evaluate_instances.py -i ${nnUNet_results}/CBCT_teeth/testset/final_ensemble -ref ${nnUNet_raw}/Dataset164_Filtered_Classes/labelsTs -np 64

python cbctseg/evaluation/evaluate_instances_with_tooth_label.py -i ${nnUNet_results}/CBCT_teeth/testset/final_single_model -ref ${nnUNet_raw}/Dataset164_Filtered_Classes/labelsTs -np 64
python cbctseg/evaluation/evaluate_instances_with_tooth_label.py -i ${nnUNet_results}/CBCT_teeth/testset/final_ensemble -ref ${nnUNet_raw}/Dataset164_Filtered_Classes/labelsTs -np 64


