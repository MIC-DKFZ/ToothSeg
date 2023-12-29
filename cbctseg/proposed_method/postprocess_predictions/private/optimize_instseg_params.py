from cbctseg.evaluation.evaluate_instances import compute_instance_only_metrics
from cbctseg.proposed_method.postprocess_predictions.border_core_to_instances import convert_all_sem_to_instance
from cbctseg.proposed_method.postprocess_predictions.resize_predictions import resample_segmentations_to_ref


def run(source_folder, small_center_threshold_default, isolated_border_as_separate_instance_threshold_default, min_instance_size, identifier):
    out = source_folder + '_' + identifier + '_instances'

    convert_all_sem_to_instance(source_folder, out, overwrite=False,
                                small_center_threshold=small_center_threshold_default,
                                isolated_border_as_separate_instance_threshold=
                                isolated_border_as_separate_instance_threshold_default,
                                num_processes=num_processes,
                                min_instance_size=min_instance_size)

    resample_segmentations_to_ref(original_seg_folder_tr, out, out + '_resized', overwrite=False, num_threads=num_processes)
    return compute_instance_only_metrics(out + '_resized', original_seg_folder_tr, num_processes=num_processes)


if __name__ == '__main__':
    original_seg_folder_tr = '/dkfz/cluster/gpu/data/OE0441/isensee/Shank_testSet/original_segs/labelsTr'
    source_folder = '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake/Dataset188_CBCTTeeth_instance_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8/fold_0/validation'
    num_processes = 128

    results = []
    small_center_threshold_default = 0.26999999999999996  # =0.3**3 * 10  # equivalent to 10 pixels at 0.3 spacing
    isolated_border_as_separate_instance_threshold_default = 0.26999999999999996  # =0.3**3 * 10  # equivalent to 10 pixels at 0.3 spacing
    min_instance_size = 20
    identifier = f'_minsize{min_instance_size}_sct{round(small_center_threshold_default, 3)}_ibsi{round(isolated_border_as_separate_instance_threshold_default, 3)}'
    results.append(
        (identifier,
         run(source_folder, small_center_threshold_default, isolated_border_as_separate_instance_threshold_default,
             min_instance_size, identifier)))

