export REFERENCE_FOLDER=/dkfz/cluster/gpu/data/OE0441/isensee/nnUNet_raw/nnUNet_raw_remake/Dataset164_Filtered_Classes/labelsTr

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset164_Filtered_Classes/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch/fold_0/validation
CUDA_VISIBLE_DEVICES=0 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset164_Filtered_Classes/nnUNetTrainer__nnUNetPlans__3d_lowres_resample_torch/fold_0/validation
CUDA_VISIBLE_DEVICES=1 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset164_Filtered_Classes/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation
CUDA_VISIBLE_DEVICES=2 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation
CUDA_VISIBLE_DEVICES=3 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation
CUDA_VISIBLE_DEVICES=4 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation
CUDA_VISIBLE_DEVICES=5 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation
CUDA_VISIBLE_DEVICES=6 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs16/fold_0/validation
CUDA_VISIBLE_DEVICES=7 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8/fold_0/validation
CUDA_VISIBLE_DEVICES=0 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances
CUDA_VISIBLE_DEVICES=0 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances
CUDA_VISIBLE_DEVICES=2 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs16/fold_0/validation_instances
CUDA_VISIBLE_DEVICES=1 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_256_bs8/fold_0/validation_instances
CUDA_VISIBLE_DEVICES=3 python ~/git_repos/radbouduni_2023_cbctteethsegmentation/cbctseg/proposed_method/postprocess_predictions/resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref ${REFERENCE_FOLDER} -np 64 &

#Dataset164_Filtered_Classes
#Dataset181_CBCTTeeth_semantic_spacing03
#Dataset186_CBCTTeeth_instance_spacing02_brd2px
#Dataset188_CBCTTeeth_instance_spacing02_brd3px
