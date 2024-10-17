from pathlib import Path

import cv2
import nibabel
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize_3d
import torch
from tqdm import tqdm

from baselines.data import transforms as T


palette = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
    (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
    (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
    (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
    (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
    (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
    (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
    (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
    (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
    (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
    (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
    (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
    (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
    (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
    (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
    (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
    (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
    (246, 0, 122), (191, 162, 208),
]


def relunet(image, labels, alpha=0.5):    
    image_raw = np.clip(image, 0, 4095) / 4095
    image_raw = image_raw.transpose((2, 1, 0))
    cv2.imwrite('visualization/mpi.png', 255 * image_raw.max(1))


    image_down = torch.nn.functional.interpolate(
        torch.from_numpy(image)[None, None].float(),
        (128, 96, 128),
    )[0, 0].numpy()
    image_down = np.clip(image_down, 0, 4095) / 4095
    image_down = image_down.transpose((2, 1, 0))
    image_down = (255 * image_down.max(1)).astype(int)
    image_down = np.tile(image_down[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_down.png', image_down)


    labels_down = torch.nn.functional.interpolate(
        torch.from_numpy(labels > 0)[None, None].float(),
        (128, 96, 128),
    )[0, 0].numpy() >= 0.5
    labels_down = labels_down.transpose((2, 1, 0))
    labels_down = labels_down.max(1)
    image_down[labels_down] = image_down[labels_down] * (1 - alpha)
    image_down[labels_down, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_down_roi.png', image_down)

    image_down = torch.nn.functional.interpolate(
        torch.from_numpy(image)[None, None].float(),
        (247, 186, 247),
    )[0, 0].numpy()
    image_down = np.clip(image_down, 0, 4095) / 4095
    image_down = image_down.transpose((2, 1, 0))
    image_np = (255 * image_down.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_resample.png', image_np)

    labels_down = np.zeros((247, 186, 247), dtype=labels.dtype)
    for label in np.unique(labels)[1:]:
        mask = torch.nn.functional.interpolate(
            input=torch.from_numpy(labels == label).float()[None, None],
            size=(247, 186, 247),
            mode='trilinear'
        )[0, 0].numpy() >= 0.5
        labels_down[mask] = label
    labels_down = labels_down.transpose((2, 1, 0))
    crop = T.VOICrop((128, 96, 128))
    crop_down = crop(intensities=image_down, labels=labels_down)
    image_np = (255 * crop_down['intensities'].max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_crop.png', image_np)


    labels_down = crop_down['labels'].max(1)
    image_np[labels_down > 0] = image_np[labels_down > 0] * (1 - alpha)
    for label in np.unique(labels_down)[1:]:
        image_np[labels_down == label] += (np.array(palette[label]) * alpha).astype(int)
    cv2.imwrite('visualization/mpi_crop_seg.png', image_np)

    image_up = torch.nn.functional.interpolate(
        torch.from_numpy(image)[None, None].float(),
        (692, 519, 692),
    )[0, 0].numpy()
    image_up = np.clip(image_up, 0, 4095) / 4095
    image_up = image_up.transpose((2, 1, 0))
    image_np = (255 * image_up.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_resample_025.png', image_np)
    
    labels_up = np.zeros((692, 519, 692), dtype=labels.dtype)
    for label in np.unique(labels)[1:]:
        mask = torch.nn.functional.interpolate(
            input=torch.from_numpy(labels == label).float()[None, None],
            size=(692, 519, 692),
            mode='trilinear'
        )[0, 0].numpy() >= 0.5
        labels_up[mask] = label
    labels_up = labels_up.transpose((2, 1, 0))

    teeth = T.ToothPatches([128, 128, 96])
    patches = teeth(intensities=image_up, labels=labels_up, instances=labels_up[None])
    image_patch = patches['intensities'][-1].max(1)
    image_np = (255 * image_patch).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_patch.png', image_np)

    labels_patch = patches['labels'][-1].max(1)
    image_np[labels_patch] = image_np[labels_patch] * (1 - alpha)
    image_np[labels_patch, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_patch_seg.png', image_np)


def liunet(image, labels, alpha=0.5):
    norm = T.HistogramNormalize()
    stats = norm(intensities=image)
    print('min:', stats['norm_min'], 'max:', stats['norm_max'])

    image_raw = np.clip(image, stats['norm_min'], stats['norm_max'])
    image_raw = (image_raw - stats['norm_min']) / (stats['norm_max'] - stats['norm_min'])
    image_raw = image_raw.transpose((2, 1, 0))
    cv2.imwrite('visualization/mpi.png', 255 * image_raw.max(1))

    image_down = torch.nn.functional.interpolate(
        torch.from_numpy(image)[None, None].float(),
        (432, 324, 432),
    )[0, 0].numpy()
    image_down = np.clip(image_down, stats['norm_min'], stats['norm_max'])
    image_down = (image_down - stats['norm_min']) / (stats['norm_max'] - stats['norm_min'])
    image_np = image_down.transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_down.png', image_np)

    labels_down = np.zeros((432, 324, 432), dtype=labels.dtype)
    for label in np.unique(labels)[1:]:
        mask = torch.nn.functional.interpolate(
            input=torch.from_numpy(labels == label).float()[None, None],
            size=(432, 324, 432),
            mode='trilinear'
        )[0, 0].numpy() >= 0.5
        labels_down[mask] = label

    labels_np = labels_down.transpose((2, 1, 0)).max(1)
    image_np[labels_np > 0] = image_np[labels_np > 0] * (1 - alpha)
    image_np[labels_np > 0, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_down_seg.png', image_np)
    
    image_np = image_down.transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    image_np[labels_np > 0] = image_np[labels_np > 0] * (1 - alpha)
    for label in np.unique(labels_np)[1:]:
        image_np[labels_np == label] += (np.array(palette[label]) * alpha).astype(int)
    cv2.imwrite('visualization/mpi_down_multi.png', image_np)

    cropper = T.RandomCrop(crop_size=[160, 96, 160], rng=np.random.default_rng(48))
    results = cropper(intensities=image_down, labels=labels_down)

    image_crop = results['intensities']
    image_np = image_crop.transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_crop.png', image_np)

    labels_crop = results['labels'].transpose((2, 1, 0)).max(1)
    image_np[labels_crop > 0] = image_np[labels_crop > 0] * (1 - alpha)
    image_np[labels_crop > 0, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_crop_seg.png', image_np)

    cropper = T.RandomCrop(crop_size=[160, 96, 160], rng=np.random.default_rng(150))
    results = cropper(intensities=image_down, labels=labels_down)

    image_crop = results['intensities']
    image_np = image_crop.transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_crop2.png', image_np)

    labels_crop = results['labels'].transpose((2, 1, 0)).max(1)
    image_np[labels_crop > 0] = image_np[labels_crop > 0] * (1 - alpha)    
    image_np[labels_crop > 0, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_crop2_seg.png', image_np)

    image_np = image_crop.transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    image_np[labels_crop > 0] = image_np[labels_crop > 0] * (1 - alpha)
    for label in np.unique(labels_crop)[1:]:
        image_np[labels_crop == label] += (np.array(palette[label]) * alpha).astype(int)
    cv2.imwrite('visualization/mpi_crop2_multi.png', image_np)


def wangnet(image, labels, alpha: float=0.5):
    image_raw = np.clip(image, 0, 4095) / 4095
    image_np = image_raw.transpose((2, 1, 0))
    cv2.imwrite('visualization/mpi.png', 255 * image_np.max(1))

    image_down = torch.nn.functional.interpolate(
        torch.from_numpy(image)[None, None].float(),
        (288, 216, 288),
    )[0, 0].numpy()
    image_down = np.clip(image_down, 0, 4095) / 4095
    image_np = image_down.transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_down.png', image_np)

    labels_down = np.zeros((288, 216, 288), dtype=labels.dtype)
    for label in np.unique(labels)[1:]:
        mask = torch.nn.functional.interpolate(
            input=torch.from_numpy(labels == label).float()[None, None],
            size=(288, 216, 288),
            mode='trilinear'
        )[0, 0].numpy() >= 0.5
        labels_down[mask] = label
        
    labels_smooth = np.zeros((288, 216, 288))
    for label in np.unique(labels_down)[1:]:
        smooth = ndimage.gaussian_filter((labels_down == label).astype(float), sigma=1.0)
        smooth = (smooth - smooth.min()) / (smooth.ptp() + 1e-6)
        labels_smooth = np.maximum(labels_smooth, smooth)
    labels_np = labels_smooth.transpose((2, 1, 0))
    labels_np = np.tile(labels_np.max(1)[:, :, None], (1, 1, 3))
    red_np = np.zeros_like(image_np)
    red_np[..., 2] = 255

    out_np = (
        image_np * (1 - labels_np / 2)
        + red_np * labels_np / 2
    ).astype(int)
    cv2.imwrite('visualization/mpi_down_seg.png', out_np)

    crop = T.VOICrop((144, 128, 128))
    crop_down = crop(intensities=image_down, labels=labels_down)
    image_np = crop_down['intensities'].transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_crop.png', image_np)
    
    labels_smooth = np.zeros((144, 128, 128))
    for label in np.unique(crop_down['labels'])[1:]:
        smooth = ndimage.gaussian_filter((crop_down['labels'] == label).astype(float), sigma=2)
        smooth = (smooth - smooth.min()) / (smooth.ptp() + 1e-6)
        labels_smooth = np.maximum(labels_smooth, smooth)
    labels_np = labels_smooth.transpose((2, 1, 0))
    labels_np = np.tile(labels_np.max(1)[:, :, None], (1, 1, 3))
    red_np = np.zeros_like(image_np)
    red_np[..., 2] = 255
    out_np = (
        image_np * (1 - labels_np / 2)
        + red_np * labels_np / 2
    ).astype(int)
    cv2.imwrite('visualization/mpi_crop_seg.png', out_np)


    labels_bandwidths = np.zeros((144, 128, 128, 3))
    for label in np.unique(crop_down['labels'])[1:]:
        bandwidth = np.column_stack(np.nonzero(crop_down['labels'] == label)).std(0)
        labels_bandwidths[crop_down['labels'] == label] = bandwidth
    labels_bandwidths = (labels_bandwidths - labels_bandwidths.min((0, 1, 2))) / labels_bandwidths.ptp((0, 1, 2))
    bandwidths_np = labels_bandwidths.transpose((2, 1, 0, 3))
    bandwidths_np = bandwidths_np.max(1)
    bandwidths_np = (
        image_np * (1 - bandwidths_np / 2)
        + 255 * bandwidths_np / 2
    ).astype(int)
    cv2.imwrite('visualization/mpi_crop_bandwidths.png', bandwidths_np)

    labels_offsets = np.zeros((144, 128, 128, 3))
    for label in np.unique(crop_down['labels'])[1:]:
        coords = np.column_stack(np.nonzero(crop_down['labels'] == label))
        centroid = coords.mean(0)
        offsets = centroid - coords
        offsets = (offsets - offsets.min(0)) / offsets.ptp(0)
        labels_offsets[crop_down['labels'] == label] = offsets
    offsets_np = labels_offsets.transpose((2, 1, 0, 3))
    offsets_np = offsets_np.max(1)
    offsets_np = (
        image_np * (1 - offsets_np / 2)
        + 255 * offsets_np / 2
    ).astype(int)
    cv2.imwrite('visualization/mpi_crop_offsets.png', offsets_np)

    image_np = crop_down['intensities'].transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    labels_np = crop_down['labels'].transpose((2, 1, 0)).max(1)
    image_np[labels_np > 0] = image_np[labels_np > 0] * (1 - alpha)
    for label in np.unique(labels_np)[1:]:
        image_np[labels_np == label] += (np.array(palette[-label.item()]) * alpha).astype(int)
    cv2.imwrite('visualization/mpi_crop_multi1.png', image_np)

    image_np = crop_down['intensities'].transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    labels_np = crop_down['labels'].transpose((2, 1, 0)).max(1)
    image_np[labels_np > 0] = image_np[labels_np > 0] * (1 - alpha)
    for label in np.unique(labels_np)[1:]:
        image_np[labels_np == label] += (np.array(palette[label]) * alpha).astype(int)
    cv2.imwrite('visualization/mpi_crop_multi2.png', image_np)

    teeth = T.ToothPatches([80, 80, 128])
    patches = teeth(intensities=image_raw, labels=labels, instances=labels[None])
    image_patch = patches['intensities'][-1].transpose((2, 1, 0)).max(1)
    image_np = (255 * image_patch).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_patch.png', image_np)

    labels_patch = patches['labels'][-1].transpose((2, 1, 0)).max(1)
    image_np[labels_patch] = image_np[labels_patch] * (1 - alpha)
    image_np[labels_patch, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_patch_seg.png', image_np)


def cuinet(image, labels, alpha: float=0.5):
    image_raw = np.clip(image, 0, 2500) / 2500
    image_np = image_raw.transpose((2, 1, 0))
    cv2.imwrite('visualization/mpi.png', 255 * image_np.max(1))
    
    image_down = torch.nn.functional.interpolate(
        torch.from_numpy(image_raw)[None, None].float(),
        (432, 324, 432),
    )[0, 0].numpy()
    image_np = image_down.transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_down.png', image_np)

    labels_down = np.zeros((432, 324, 432), dtype=labels.dtype)
    for label in np.unique(labels)[1:]:
        mask = torch.nn.functional.interpolate(
            input=torch.from_numpy(labels == label).float()[None, None],
            size=(432, 324, 432),
            mode='trilinear'
        )[0, 0].numpy() >= 0.5
        labels_down[mask] = label
    labels_np = labels_down.transpose((2, 1, 0)).max(1)
    image_np[labels_np > 0] = image_np[labels_np > 0] * (1 - alpha)
    image_np[labels_np > 0, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_down_seg.png', image_np)

    cropper = T.RandomCrop(crop_size=[256, 256, 256], rng=np.random.default_rng(48))
    results = cropper(intensities=image_down, labels=labels_down)
    
    image_crop = results['intensities']
    image_np = image_crop.transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_crop.png', image_np)

    labels_crop = results['labels'].transpose((2, 1, 0)).max(1)
    image_np[labels_crop > 0] = image_np[labels_crop > 0] * (1 - alpha)
    image_np[labels_crop > 0, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_crop_seg.png', image_np)

    crop = T.VOICrop((256, 256, 256))
    crop_down = crop(intensities=image_down, labels=labels_down)
    image_np = (255 * crop_down['intensities'].transpose((2, 1, 0)).max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    cv2.imwrite('visualization/mpi_voi.png', image_np)

    labels_np = crop_down['labels'].transpose((2, 1, 0)).max(1)
    image_np[labels_np > 0] = image_np[labels_np > 0] * (1 - alpha)
    image_np[labels_np > 0, 2] += int(alpha * 255)
    cv2.imwrite('visualization/mpi_voi_seg.png', image_np)

    labels_offsets = np.zeros((256, 256, 256, 3))
    for label in np.unique(crop_down['labels'])[1:]:
        coords = np.column_stack(np.nonzero(crop_down['labels'] == label))
        centroid = coords.mean(0)
        offsets = centroid - coords
        offsets = (offsets - offsets.min(0)) / offsets.ptp(0)
        labels_offsets[crop_down['labels'] == label] = offsets
    offsets_np = labels_offsets.transpose((2, 1, 0, 3))
    offsets_np = offsets_np.max(1)
    offsets_np = (
        image_np * (1 - offsets_np / 2)
        + 255 * offsets_np / 2
    ).astype(int)
    cv2.imwrite('visualization/mpi_voi_centroid_offsets.png', offsets_np)

    transform = T.ToothSkeletonOffsets()
    labels_offsets = transform(labels=crop_down['labels'])
    labels_offsets = labels_offsets['offsets'].transpose((1, 2, 3, 0))
    for label in np.unique(crop_down['labels'])[1:]:
        offsets = labels_offsets[crop_down['labels'] == label]
        offsets = (offsets - offsets.min(0)) / offsets.ptp(0)
        labels_offsets[crop_down['labels'] == label] = offsets
    offsets_np = labels_offsets.transpose((2, 1, 0, 3)).max(1)
    offsets_np = (
        image_np * (1 - offsets_np / 2)
        + 255 * offsets_np / 2
    ).astype(int)
    cv2.imwrite('visualization/mpi_voi_skeleton_offsets.png', offsets_np)

    image_np = crop_down['intensities'].transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    labels_np = crop_down['labels'].transpose((2, 1, 0)).max(1)
    image_np[labels_np > 0] = image_np[labels_np > 0] * (1 - alpha)
    for label in np.unique(labels_np)[1:]:
        image_np[labels_np == label] += (np.array(palette[-label.item()]) * alpha).astype(int)
    cv2.imwrite('visualization/mpi_voi_instances.png', image_np)

    image_np = crop_down['intensities'].transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))

    keypoints = np.zeros((256, 256, 3))
    ratio = np.zeros((256, 256, 256))
    labels_np = crop_down['labels'].transpose((2, 1, 0))
    for label in tqdm(np.unique(labels_np)[1:]):
        centroid = np.column_stack(np.nonzero(labels_np == label)).mean(0).astype(int)

        keypoint = np.zeros_like(labels_np).astype(float)
        keypoint[tuple(centroid)] = 1.0
        keypoint = ndimage.gaussian_filter(keypoint, sigma=2.0)
        keypoint = (keypoint - keypoint.min()) / keypoint.ptp()

        fg = keypoint.max(1) > ratio.max(1)
        keypoints[fg] = palette[-label.item()]
        ratio = np.maximum(ratio, keypoint)
    ratio = np.tile(ratio[..., None], (1, 1, 1, 3)).max(1)
    out_np = (
        image_np * (1 - ratio)
        + keypoints * ratio
    ).astype(int)
    cv2.imwrite('visualization/mpi_voi_centroids.png', out_np)


    image_np = crop_down['intensities'].transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))

    keypoints = np.zeros((256, 256, 3))
    ratio = np.zeros((256, 256, 256))
    labels_np = crop_down['labels'].transpose((2, 1, 0))
    for label in tqdm(np.unique(labels_np)[1:]):
        keypoint = skeletonize_3d(labels_np == label) > 0
        keypoint = ndimage.gaussian_filter(keypoint.astype(float), sigma=2.0)
        keypoint = (keypoint - keypoint.min()) / keypoint.ptp()

        fg = keypoint.max(1) > ratio.max(1)
        keypoints[fg] = palette[-label.item()]
        ratio = np.maximum(ratio, keypoint)
    ratio = np.tile(ratio[..., None], (1, 1, 1, 3)).max(1)
    out_np = (
        image_np * (1 - ratio)
        + keypoints * ratio
    ).astype(int)
    cv2.imwrite('visualization/mpi_voi_skeletons.png', out_np)

    boundaries = T.ToothBoundaries()(labels=crop_down['labels'])['boundaries']
    keypoints = T.ToothRootKeypoints()(labels=crop_down['labels'])['keypoints']
    instances = np.zeros((3, 256, 256, 256)).astype(int)
    for label in tqdm(np.unique(crop_down['labels'])[1:]):
        mask = crop_down['labels'] == label
        instances[0, mask] = label

        centroid = np.column_stack(np.nonzero(mask)).mean(0).astype(int)
        instances[1, *tuple(centroid)] = label

        skeleton = skeletonize_3d(mask) > 0
        instances[2, skeleton] = label

    patches = T.ToothPatches(patch_size=[96, 96, 96], sigma=2.0)
    results = patches(
        intensities=crop_down['intensities'],
        labels=crop_down['labels'],
        instances=instances,
        boundaries=boundaries,
        keypoints=keypoints,
    )

    image_np = results['intensities'][-1].transpose((2, 1, 0))
    image_np = (255 * image_np.max(1)).astype(int)
    image_np = np.tile(image_np[..., None], (1, 1, 3))
    
    red_np = np.zeros_like(image_np)
    red_np[..., 2] = 255
    skeleton_np = results['instances'][-1][2].transpose((2, 1, 0))
    skeleton_np = np.tile(skeleton_np.max(1)[..., None], (1, 1, 3))

    
    green_np = np.zeros_like(image_np)
    green_np[..., 1] = 255
    centroid_np = results['instances'][-1][1].transpose((2, 1, 0))
    centroid_np = np.tile(centroid_np.max(1)[..., None], (1, 1, 3))

    out_np = (
        image_np * (1 - skeleton_np)
        + red_np * skeleton_np
    ).astype(int)
    out_np = (
        out_np * (1 - centroid_np)
        + green_np * centroid_np
    ).astype(int)
    cv2.imwrite('visualization/mpi_patch.png', out_np)

    boundaries_np = results['boundaries'][-1].transpose((2, 1, 0))
    boundaries_np = boundaries_np.mean(1)
    boundaries_np = (boundaries_np - boundaries_np.min()) / boundaries_np.ptp()
    boundaries_np = np.tile(boundaries_np[..., None], (1, 1, 3))
    out_np = (
        image_np * (1 - boundaries_np)
        + red_np * boundaries_np
    ).astype(int)
    cv2.imwrite('visualization/mpi_patch_boundary.png', out_np)

    keypoints_np = results['keypoints'][-1].transpose((2, 1, 0))
    keypoints_np = np.tile(keypoints_np.max(1)[..., None], (1, 1, 3))
    out_np = (
        image_np * (1 - keypoints_np)
        + red_np * keypoints_np
    ).astype(int)
    cv2.imwrite('visualization/mpi_patch_keypoints.png', out_np)

    labels_np = results['labels'][-1].transpose((2, 1, 0))
    labels_np = np.tile(labels_np.max(1)[..., None], (1, 1, 3))
    out_np = (
        image_np * (1 - labels_np / 2)
        + red_np * labels_np / 2
    ).astype(int)
    cv2.imwrite('visualization/mpi_patch_seg.png', out_np)

    color_np = np.array(palette[results['unique_labels'][-1]])
    color_np = np.tile(color_np[None, None], (96, 96, 1))
    out_np = (
        image_np * (1 - labels_np / 2)
        + color_np * labels_np / 2
    ).astype(int)
    cv2.imwrite('visualization/mpi_patch_multi.png', out_np)


if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/tooth_segmentation/data/Dataset164_Filtered_Classes/imagesTr/')

    img_nii = nibabel.load(root / 'lenka-probable-hornet_0000.nii.gz')
    image = np.asarray(img_nii.dataobj)
    image = image[96:-96, :432]

    labels_nii = nibabel.load(root.parent / 'labelsTr' / 'lenka-probable-hornet.nii.gz')
    labels = np.asarray(labels_nii.dataobj)
    labels = labels[96:-96, :432]

    # relunet(image, labels)
    # liunet(image, labels)
    # wangnet(image, labels)
    cuinet(image, labels)
