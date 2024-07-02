# Dataset Prep
1. Convert dataset to nifti with `convert_to_nifti.py`
2. Remove NA labels with `remove_na_classes.py`
3. Create teeth only dataset with `teeth_only.py`. This will be our semantic datasets
4. Create instance segmentation dataset with `prepare_dataset/instance_segmentation.py`:
   - We need to pick a target spacing here already because the border-core representation is sensitive to that. 
   When using toothfairy for our paper we keep things consistent, so 0.2 it is even though all tooth fairy data is 0.3. 
   Changing target spacing here would be test set overfitting. When participating in the challenge we can pick 0.3 because 0.3 makes sense
   - 



# nnU-Net planning and preprocessing
