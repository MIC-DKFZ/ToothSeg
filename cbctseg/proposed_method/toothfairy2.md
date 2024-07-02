# Dataset Prep
1. Convert dataset to nifti with `convert_to_nifti.py`
2. Remove NA labels with `remove_na_classes.py`
3. Create teeth only dataset with `teeth_only.py`. This will be our semantic dataset
4. Run `prepare_dataset/toothfairy2.py` to create the instance seg dataset. This requires a GPU!
5. 

# nnU-Net planning and preprocessing
