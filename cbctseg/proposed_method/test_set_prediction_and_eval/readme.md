# Run test set predictions

1. Resize the test set for the instance segmentation (border-core) prediction via `resize_test_set.py`. This is not needed for semantic seg
2. Run the predictions. Take inspiration from `predict_test_set.sh`
     1. Semantic segmentations will already be in the correct spacing (same as original images)
     2. Border-core instance segmentations are in their 0.2x0.2x0.2 spacing (needed for proper conversion) and must be processed further
3. Convert the border-core prediction into instances with `cbctseg/proposed_method/postprocess_predictions/border_core_to_instances.py`
4. Resize instances to original spacing with `cbctseg/proposed_method/postprocess_predictions/resize_predictions.py`
5. Now we can assign the predicted instances the correct tooth labels (as predicted by the semantic segmentation model) with `cbctseg/proposed_method/postprocess_predictions/assign_tooth_labels.py`

# Evaluate test set

1. Evaluate the quality of the instances (regardless of tooth label) with `cbctseg/evaluation/evaluate_instances.py`
2. Evaluate the quality of the predictions (instances AND correct tooth label) with `cbctseg/evaluation/evaluate_instances_with_tooth_label.py`