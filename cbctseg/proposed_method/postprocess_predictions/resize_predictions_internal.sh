REFERENCE_FOLDER=/dkfz/cluster/gpu/data/OE0441/isensee/Shank_testSet/original_segs/labelsTr

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset185_CBCTTeeth_instance_spacing05_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset184_CBCTTeeth_instance_spacing03_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset183_CBCTTeeth_semantic_spacing02/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset182_CBCTTeeth_semantic_spacing05/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset181_CBCTTeeth_semantic_spacing03/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset186_CBCTTeeth_instance_spacing02_brd2px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset187_CBCTTeeth_instance_spacing03_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_128/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128

INPUT_FOLDER=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation_instances,
python resize_predictions.py -i $INPUT_FOLDER -o ${INPUT_FOLDER}_resized -ref $REFERENCE_FOLDER -np 128