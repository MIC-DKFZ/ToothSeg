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