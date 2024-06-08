Our dataset preparation already did resampling. This is necessary because border/core needs a consistent border size. 
Converting to border core and then letting nnU-Net do the resampling would lead to inconsistent border thicknesses. 
So that's da wae.

Evaluation should (as always) be done in the original image size. So we need to resize all images back to the original 
image size before doing anything else with them. This is what the scripts in this folder are for.

Instructions:

1. Semantic segmentation
   - resize to original sizes with resize_predictions.py
2. Border-Core (instance) segmentations
   - first convert border-core representation to instances with border_core_to_instances.py
   - then convert back to original sizes with resize_predictions.py
3. Bring it all together with assign_tooth_labels.py