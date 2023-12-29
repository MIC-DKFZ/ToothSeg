INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation/
python border_core_to_instances.py -i ${INPUT_FOLDER} -o ${INPUT_FOLDER}_instances -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation
python border_core_to_instances.py -i ${INPUT_FOLDER} -o ${INPUT_FOLDER}_instances -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs16/fold_0/validation
python border_core_to_instances.py -i ${INPUT_FOLDER} -o ${INPUT_FOLDER}_instances -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_256_bs8/fold_0/validation
python border_core_to_instances.py -i ${INPUT_FOLDER} -o ${INPUT_FOLDER}_instances -np 128



#Dataset164_Filtered_Classes
#Dataset181_CBCTTeeth_semantic_spacing03
#Dataset186_CBCTTeeth_instance_spacing02_brd2px
#Dataset188_CBCTTeeth_instance_spacing02_brd3px