import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import nibabel
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize_3d
from skimage.segmentation import find_boundaries
import torch
from torchtyping import TensorType

from baselines.models.cluster import fast_search_cluster


class Compose:

    def __init__(
        self,
        *transforms: List[Callable[..., Dict[str, Any]]],
    ):
        self.transforms = list(transforms)

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        for t in self.transforms:
            data_dict = t(**data_dict)
        
        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            *[
                '    ' + repr(t).replace('\n', '\n    ') + ','
                for t in self.transforms
            ],
            ')',
        ])


class RegularSpacing:

    def __init__(
        self,
        spacing: Optional[Union[float, ArrayLike]],
    ) -> None:
        if isinstance(spacing, float):
            spacing = [spacing]*3

        if spacing is not None:
            spacing = np.array(spacing)

        self.spacing = spacing

    def __call__(
        self,
        intensities: NDArray[Any],
        spacing: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no interpolation is applied
        if self.spacing is None:
            data_dict['intensities'] = intensities
            data_dict['spacing'] = spacing

            return data_dict

        # compute how much bigger results should be
        zoom = spacing / self.spacing
        
        # immediately return if zoom is not necessary
        if np.allclose(zoom, 1):
            data_dict['intensities'] = intensities
            data_dict['spacing'] = self.spacing

            return data_dict

        # interpolate intensities volume to given voxel spacing
        out_size = np.ceil(zoom * intensities.shape).astype(int)
        out_size = tuple(out_size.tolist())
        data_dict['intensities'] = torch.nn.functional.interpolate(
            input=torch.from_numpy(intensities).float()[None, None],
            size=out_size,
            mode='trilinear'
        )[0, 0].trunc().short().numpy()
        
        # interpolate mandible segmentation to given voxel spacing
        if 'mandible' in data_dict:
            data_dict['mandible'] = torch.nn.functional.interpolate(
                input=torch.from_numpy(data_dict['mandible']).float()[None, None],
                size=out_size,
                mode='trilinear'
            )[0, 0].numpy() >= 0.5

        # interpolate labels volume to given voxel spacing
        if 'labels' in data_dict:
            labels = np.zeros(out_size, dtype=data_dict['labels'].dtype)
            for label in np.unique(data_dict['labels'])[1:]:
                mask = torch.nn.functional.interpolate(
                    input=torch.from_numpy(data_dict['labels'] == label).float()[None, None],
                    size=out_size,
                    mode='trilinear'
                )[0, 0].numpy() >= 0.5
                labels[mask] = label
            data_dict['labels'] = labels

        # interpolate instances volumes to given voxel spacing
        if 'instances' in data_dict:
            representations = []
            for i in range(data_dict['instances'].shape[0]):
                instances = np.zeros(out_size, dtype=data_dict['instances'].dtype)
                for label in np.unique(data_dict['instances'][i])[1:]:
                    mask = torch.nn.functional.interpolate(
                        input=torch.from_numpy(data_dict['instances'][i] == label).float()[None, None],
                        size=out_size,
                        mode='trilinear'
                    )[0, 0].numpy() >= 0.5
                    instances[mask] = label
                representations.append(instances)
            data_dict['instances'] = np.stack(representations)

        # update voxel spacing accordingly
        data_dict['spacing'] = self.spacing

        # determine affine transformation from input to result
        affine = np.eye(4)
        affine[np.diag_indices(3)] = zoom
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    spacing={self.spacing},',
            ')',
        ])
    

class Downsample:

    def __init__(
        self,
        crop_size: Union[int, ArrayLike],
        **kwargs,
    ) -> None:
        if isinstance(crop_size, int):
            crop_size = (crop_size,)*3

        self.crop_size = tuple(crop_size)

    def __call__(
        self,
        intensities: NDArray[Any],
        shape: NDArray[Any],
        **data_dict: Dict[str, Any],
    ):
        # interpolate intensities volume to given voxel spacing
        data_dict['intensities'] = torch.nn.functional.interpolate(
            input=torch.from_numpy(intensities).float()[None, None],
            size=self.crop_size,
            mode='trilinear'
        )[0, 0].trunc().short().numpy()
        
        # interpolate mandible segmentation to given voxel spacing
        if 'mandible' in data_dict:
            data_dict['mandible'] = torch.nn.functional.interpolate(
                input=torch.from_numpy(data_dict['mandible']).float()[None, None],
                size=self.crop_size,
                mode='trilinear'
            )[0, 0].numpy() >= 0.5

        # interpolate labels volume to given voxel spacing
        if 'labels' in data_dict:
            labels = np.zeros(self.crop_size, dtype=data_dict['labels'].dtype)
            for label in np.unique(data_dict['labels'])[1:]:
                mask = torch.nn.functional.interpolate(
                    input=torch.from_numpy(data_dict['labels'] == label).float()[None, None],
                    size=self.crop_size,
                    mode='trilinear'
                )[0, 0].numpy() >= 0.5
                labels[mask] = label
            data_dict['labels'] = labels

        if 'instances' in data_dict:
            representations = []
            for i in data_dict['instances'].shape[0]:
                instances = np.zeros(self.crop_size, dtype=data_dict['instances'].dtype)
                for label in np.unique(data_dict['instances'][i])[1:]:
                    mask = torch.nn.functional.interpolate(
                        input=torch.from_numpy(data_dict['instances'][i] == label).float()[None, None],
                        size=self.crop_size,
                        mode='trilinear'
                    )[0, 0].numpy() >= 0.5
                    instances[mask] = label
                representations.append(instances)
            data_dict['instances'] = np.stack(representations)

        # update voxel spacing accordingly
        factor = shape / self.crop_size
        data_dict['shape'] = shape

        # determine affine transformation from input to result
        affine = np.eye(4)
        affine[np.diag_indices(3)] = 1 / factor
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    crop_size ={self.crop_size},',
            ')',
        ])


class NaturalHeadPositionOrient:

    def __call__(
        self,
        orientation: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # reorient volumes to standard basis
        for key in ['intensities', 'mandible', 'labels', 'instances']:
            if key not in data_dict:
                continue

            ornt = np.concatenate((
                np.array([[0, 1]]),
                np.column_stack((orientation[:, 0] + 1, orientation[:, 1])),
            )) if data_dict[key].ndim > 3 else orientation
            data_dict[key] = nibabel.apply_orientation(
                arr=data_dict[key],
                ornt=ornt,
            )

            shape = data_dict[key].shape[-3:]
        
        # update orientation to identity
        data_dict['orientation'] = nibabel.io_orientation(affine=np.eye(4))

        # determine affine transformation from input to result
        inv_affine = nibabel.orientations.inv_ornt_aff(
            ornt=orientation,
            shape=shape,
        )
        affine = np.linalg.inv(inv_affine)
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
    

class Crop:

    def __init__(
        self,
        crop_size: Union[Tuple[int, int, int], int],
        **kwargs,
    ):
        if isinstance(crop_size, int):
            crop_size = (crop_size,)*3
        self.crop_size = np.array(crop_size)

        self.pad = Pad(crop_size)

    def start_coords(
        self,
        **data_dict: Dict[str, Any],
    ) -> ArrayLike:
        raise NotImplementedError('Please use child class.')

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict = self.pad(**data_dict)

        start_coords = self.start_coords(**data_dict)
        slices = tuple([slice(l, l + dim) for l, dim in zip(start_coords, self.crop_size)])

        # crop volumes given slices
        for key in ['intensities', 'mandible', 'instances', 'labels', 'boundaries', 'offsets', 'keypoints']:
            if key not in data_dict:
                continue
            
            slices_ = (slice(None),)*(data_dict[key].ndim - 3) + slices
            data_dict[key] = data_dict[key][slices_]

        # determine affine transformation from source to crop
        affine = np.eye(4)
        affine[:3, 3] -= [slc.start for slc in slices]
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))
        
        return data_dict
        
    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    crop_size ={self.crop_size},',
            ')',
        ])


class VOICrop(Crop):            
    
    def _fit_bbox(
        self,
        slices,
        **data_dict,
    ):
        # crop or expand bounding box to exactly crop_size
        out = ()
        for slc, dim, crop_size in zip(slices, data_dict['intensities'].shape, self.crop_size):
            diff = crop_size - (slc.stop - slc.start)
            diff = diff // 2, diff // 2 + diff % 2
            slc = slice(slc.start - diff[0], slc.stop + diff[1])
            diff = dim - min(slc.start, 0) - max(dim, slc.stop)
            slc = slice(slc.start + diff, slc.stop + diff)
            out += (slc,)
            
        return out

    def start_coords(
        self,
        **data_dict: Dict[str, Any],
    ) -> ArrayLike:
        if np.any(data_dict['labels']):
            foreground_mask = data_dict['labels'] > 0
        else:
            foreground_mask = data_dict['mandible'] > 0
        
        slices = ndimage.find_objects(foreground_mask)[0]
        slices = self._fit_bbox(slices, **data_dict)

        return np.array([slc.start for slc in slices])
    

class Pad:

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            crop_size = (crop_size,)*3
        
        self.crop_size = crop_size

    def __call__(
        self,
        intensities,
        **data_dict: Dict[str, Any],
    ):
        # pad CBCT if it is too small
        pads = [max(c - d, 0) for c, d in zip(self.crop_size, intensities.shape)]
        pads = [(pad // 2, pad // 2 + pad % 2) for pad in pads]
        data_dict['intensities'] = np.pad(intensities, pads, constant_values=-1000)
        for key in ['mandible', 'instances', 'labels', 'boundaries', 'offsets', 'keypoints']:
            if key not in data_dict:
                continue

            pads_ = [(0,0)]*(data_dict[key].ndim - 3) + pads
            data_dict[key] = np.pad(data_dict[key], pads_)
        
        affine = np.eye(4)
        affine[:3, 3] += [diff[0] for diff in pads]
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    crop_size={self.crop_size},',
            ')',
        ])


class ToothBoundaries:

    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not np.any(data_dict['labels']):
            data_dict['boundaries'] = data_dict['labels'].astype(float)
            return data_dict

        boundaries = find_boundaries(data_dict['labels'])

        data_dict['boundaries'] = boundaries
        
        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ToothRootKeypoints:

    def __init__(self, **kwargs):
        pass

    def find_roots_components(
        self, labels, label,
    ):   
        tooth_mask = ndimage.binary_opening(
            labels == label,
            ndimage.generate_binary_structure(3, 1),
            iterations=1,
        )

        centroid = np.column_stack(np.nonzero(tooth_mask)).mean(0).astype(int)

        roots_mask = tooth_mask.copy()
        if label < 30:
            roots_mask[..., centroid[2] - 12:] = False
        else:
            roots_mask[..., :centroid[2] + 12] = False

        roots_coords = np.column_stack(np.nonzero(roots_mask))
        peak_voxels, cluster_idxs = fast_search_cluster(
            roots_coords[:, :-1],
            density_threshold=8,
            distance_threshold=7.07,  # sqrt(5^2 + 5^2)
        )

        root_coords = np.zeros((0, 3), dtype=int)
        for i in range(peak_voxels.shape[0]):
            coords = roots_coords[cluster_idxs == (i + 1)]
            dists = np.linalg.norm(coords - centroid, axis=-1)
            root_coord = coords[dists.argmax()]
            
            root_coords = np.concatenate((root_coords, [root_coord]))
        
        out = np.zeros_like(tooth_mask)
        out[tuple(root_coords.T)] = True

        return out

    def __call__(
        self,
        labels,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        unique_labels = np.unique(labels)[1:]
        out = np.zeros_like(labels)
        for label in unique_labels:
            keypoints = self.find_roots_components(labels, label)
            out[keypoints] = label

        data_dict['labels'] = labels
        data_dict['keypoints'] = out

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
    

class ToothCentroidOffsets:

    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if 'labels' not in data_dict:
            return data_dict
        
        offsets = np.zeros((3,) + data_dict['labels'].shape, dtype=float)
        unique_labels = np.unique(data_dict['labels'])[1:]
        for label in unique_labels:
            mask = data_dict['labels'] == label
            mask = np.tile(mask[None], (3, 1, 1, 1))
            mask_coords = np.column_stack(np.nonzero(mask))

            centroid = mask_coords.mean(axis=0)[1:]
            centroid_offsets = centroid - mask_coords[:mask.sum() // 3, 1:]

            offsets[tuple(mask_coords.T)] = centroid_offsets.T.flatten()

        data_dict['offsets'] = offsets

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ToothSkeletonOffsets:

    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if 'labels' not in data_dict:
            return data_dict
        
        offsets = np.zeros((3,) + data_dict['labels'].shape, dtype=float)
        unique_labels = np.unique(data_dict['labels'])[1:]
        for label in unique_labels:
            mask = data_dict['labels'] == label
            mask = np.tile(mask[None], (3, 1, 1, 1))
            mask_coords = np.column_stack(np.nonzero(mask))

            skeleton = skeletonize_3d(mask[0]) > 0
            skeleton_coords = np.column_stack(np.nonzero(skeleton))

            # use tooth centroid if skeleton is empty
            if skeleton_coords.shape[0] == 0:
                skeleton_coords = mask_coords.mean(axis=0, keepdims=True)[:, 1:]

            diffs = skeleton_coords[None] - mask_coords[:mask.sum() // 3, None, 1:]
            min_idxs = np.linalg.norm(diffs, axis=-1).argmin(axis=1)
            skeleton_offsets = diffs[np.arange(min_idxs.shape[0]), min_idxs]

            offsets[tuple(mask_coords.T)] = skeleton_offsets.T.flatten()

        data_dict['offsets'] = offsets

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
    

class RandomPatches:

    def __init__(
        self,
        max_patches: int,
        rng: Optional[np.random.Generator]=None,
        **kwargs,
    ):
        self.max_patches = max_patches
        self.rng = rng if rng is not None else rng

    def __call__(self, **data_dict):
        idxs = self.rng.choice(
            a=data_dict['unique_labels'].shape[0],
            size=min(data_dict['unique_labels'].shape[0], self.max_patches),
            replace=False,
        )

        for key in ['intensities', 'labels', 'instances', 'boundaries', 'keypoints', 'patches_idxs', 'unique_labels']:
            if (
                key not in data_dict
                or data_dict[key].shape[0] != data_dict['unique_labels'].shape[0]
            ):
                continue

            data_dict[key] = data_dict[key][idxs]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_patches={self.max_patches},',
            ')',
        ])


class MatchPredToGTInstances:

    def __init__(
        self,
        max_dist_mm: float=7.28,  # min=7.27, max 7.28
        **kwargs,
    ):
        self.max_dist_mm = max_dist_mm        

    def __call__(
        self,
        labels,
        instances,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        unique, inverse = np.unique(instances, return_inverse=True)
        unique_labels = np.arange(1, unique.shape[0])
        instances = inverse.reshape(instances.shape)

        if labels.max() == 0:
            data_dict['labels'] = labels
            data_dict['instances'] = np.where(instances > 0, 100 + instances, 0)
            return data_dict

        inst_centroids = np.zeros((0, 3))
        for label in unique_labels:
            centroid = np.column_stack(np.nonzero(instances[0] == label)).mean(0)
            inst_centroids = np.concatenate((inst_centroids, [centroid]))

        fdis = np.unique(labels)[1:]
        fdi_centroids = np.zeros((0, 3))
        for label in fdis:
            centroid = np.column_stack(np.nonzero(labels == label)).mean(0)
            fdi_centroids = np.concatenate((fdi_centroids, [centroid]))

        dists = np.linalg.norm(fdi_centroids[None] - inst_centroids[:, None], axis=-1)
        assert np.all(data_dict['spacing'][0] == data_dict['spacing'])
        max_dist_voxel = self.max_dist_mm / data_dict['spacing'][0]
        keep = dists.min(axis=1) < max_dist_voxel
        if not np.all(keep):
            print('Filtered at thr', max_dist_voxel, data_dict['scan_file'], dists.min(axis=1))

        fdi_map = np.concatenate((
            [0], fdis[dists.argmin(axis=1)],
        ))
        fdi_map[1:][~keep] = 100 + np.nonzero(~keep)[0]

        data_dict['labels'] = labels
        try:
            data_dict['instances'] = fdi_map[instances]
        except:
            print('error', data_dict['scan_file'])
            exit()

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_dist (mm)={self.max_dist_mm},',
            ')',
        ])


class ToothPatches:

    def __init__(
        self,
        patch_size: Union[Tuple[int, int, int], int],
        sigma: float=1.0,
        **kwargs,
    ):
        if isinstance(patch_size, int):
            patch_size = (patch_size,)*3
        self.patch_size = np.array(patch_size)
        self.sigma = sigma

        self.pad = Pad(patch_size)

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict = self.pad(**data_dict)
        intensities = data_dict['intensities']
        instances = data_dict['instances']
        
        patches_idxs = np.zeros((0, 3, 2), dtype=int)
        cbct_patches = np.zeros((0, *self.patch_size), dtype=intensities.dtype)
        instances_patches = np.zeros((0, instances.shape[0], *self.patch_size), dtype=np.float32)
        
        labels_patches = np.zeros((0, *self.patch_size), dtype=bool)
        boundaries_patches = np.zeros((0, *self.patch_size), dtype=np.float32)
        keypoints_patches = np.zeros((0, *self.patch_size), dtype=np.float32)

        instance_labels = np.unique(instances)[1:]
        for label in instance_labels.copy():
            # find smallest bounding box that encapsulates large objects
            try:
                tooth_slices = ndimage.find_objects(instances[-1] == label)[0]
            except IndexError:
                label_idx = instance_labels.tolist().index(label)
                instance_labels = np.concatenate((
                    instance_labels[:label_idx], instance_labels[label_idx + 1:],
                ))
                continue

            # crop or expand bounding box to exactly crop_size
            patch_slices = ()
            for patch_size, dim, slc in zip(
                self.patch_size, intensities.shape, tooth_slices,
            ):
                diff = patch_size - (slc.stop - slc.start)
                diff = (diff // 2, diff // 2 + diff % 2)
                slc = slice(slc.start - diff[0], slc.stop + diff[1])
                diff = dim - min(slc.start, 0) - max(dim, slc.stop)
                slc = slice(slc.start + diff, slc.stop + diff)
                patch_slices += (slc,)

            patch_idxs = np.array([
                [slc.start, slc.stop] for slc in patch_slices
            ])

            patches_idxs = np.concatenate((patches_idxs, [patch_idxs]))
            cbct_patches = np.concatenate((cbct_patches, [intensities[patch_slices]]))
            
            if 'labels' in data_dict:
                labels_patch = data_dict['labels'][patch_slices] == label
            else:
                labels_patch = np.zeros_like(cbct_patches[-1]).astype(bool)
            labels_patches = np.concatenate((labels_patches, [labels_patch]))

            instance = instances[(slice(None),) + patch_slices] == label
            instance = ndimage.gaussian_filter(
                instance.astype(float), sigma=self.sigma, axes=(-3, -2, -1),
            )
            instance = instance - instance.min(axis=(-3, -2, -1), keepdims=True)
            instance = instance / (instance.ptp(axis=(-3, -2, -1), keepdims=True) + 1e-6)
            instances_patches = np.concatenate((instances_patches, [instance]))

            if 'boundaries' in data_dict:
                tooth_mask = ndimage.binary_dilation(labels_patch, ndimage.generate_binary_structure(3, 1))
                boundary = data_dict['boundaries'][patch_slices] * tooth_mask
                boundary = ndimage.gaussian_filter(boundary.astype(float), sigma=self.sigma)
                boundary = (boundary - boundary.min()) / (boundary.ptp() + 1e-6)
                boundaries_patches = np.concatenate((boundaries_patches, [boundary]))

            if 'keypoints' in data_dict:
                keypoints = data_dict['keypoints'][patch_slices] == label
                keypoints = ndimage.gaussian_filter(keypoints.astype(float), sigma=self.sigma)
                keypoints = (keypoints - keypoints.min()) / (keypoints.ptp() + 1e-6)
                keypoints_patches = np.concatenate((keypoints_patches, [keypoints]))

        data_dict['patches_idxs'] = patches_idxs
        data_dict['intensities'] = cbct_patches
        data_dict['labels'] = labels_patches

        data_dict['instances'] = instances_patches
        data_dict['boundaries'] = boundaries_patches
        data_dict['keypoints'] = keypoints_patches

        data_dict['unique_labels'] = instance_labels

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    patch_size={self.patch_size},',
            f'    sigma={self.sigma},',
            ')',
        ])


class RandomCrop(Crop):

    def __init__(
        self,
        rng: Optional[np.random.Generator]=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.rng = rng if rng is not None else np.random.default_rng()

    def start_coords(
        self,
        **data_dict: Dict[str, Any],
    ) -> ArrayLike:
        margin = data_dict['intensities'].shape - self.crop_size
        margin = np.maximum(margin, 0)
        start_coords = self.rng.integers(margin, endpoint=True)

        return start_coords


class RandomTeethCrop(RandomCrop):

    def start_coords(
        self,
        **data_dict: Dict[str, Any],
    ) -> ArrayLike:
        # take a random crop during prediction and for edentulous patients
        if 'labels' not in data_dict or not np.any(data_dict['labels']):
            return super().start_coords(**data_dict)

        slices = ndimage.find_objects(data_dict['labels'] > 0)[0]
        start_coords = []
        for slc, dim, size in zip(slices, data_dict['labels'].shape, self.crop_size):
            if slc.stop - slc.start > size:
                logging.warn(f"""
                    RandomTeethCrop cannot crop entire dentition.
                    scan_file={Path(data_dict['scan_file']).name},
                    crop_size={self.crop_size},
                    teeth_slices={slices}.
                """)
                start_coords.append(slc.start)
                continue
            
            margin = max(slc.stop - size, 0), min(dim - size, slc.start)
            start_idx = self.rng.integers(*margin, endpoint=True)
            start_coords.append(start_idx)

        return np.array(start_coords)


class RandomXAxisFlip:

    def __init__(
        self,
        p: float=0.5,
        rng: Optional[np.random.Generator]=None,
        **kwargs,
    ) -> None:
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.rng.random() >= self.p:
            return data_dict

        # flip x-axis of volumes
        for key in ['intensities', 'mandible', 'labels', 'instances', 'boundaries', 'offsets', 'keypoints']:
            if key not in data_dict:
                continue
            
            data_dict[key] = np.flip(data_dict[key], -3)

        if 'offsets' in data_dict:
            data_dict['offsets'][0] *= -1

        if (
            'labels' in data_dict
            and data_dict['labels'].size
            and data_dict['labels'].max() > 1
        ):
            labels = data_dict['labels'].copy()
            labels[labels > 0] = np.where(
                (labels[labels > 0] // 10) % 2 == 0,
                labels[labels > 0] - 10,
                labels[labels > 0] + 10,
            )
            data_dict['labels'] = labels

        if 'unique_labels' in data_dict:
            unique_labels = data_dict['unique_labels'].copy()
            unique_labels = np.where(
                (unique_labels // 10) % 2 == 0,
                unique_labels - 10,
                unique_labels + 10,
            )
            data_dict['unique_labels'] = unique_labels

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    p={self.p},',
            ')',
        ])


class RandomMaskOut:

    def __init__(
        self,
        mask_size: Union[Tuple[int, int, int], int]=12,
        max_masks: int=16,
        rng: Optional[np.random.Generator]=None,
        **kwargs,
    ) -> None:
        if isinstance(mask_size, int):
            mask_size = (mask_size,)*3

        self.mask_size = np.array(mask_size)
        self.max_masks = max_masks
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        cval = data_dict['norm_mean'] if 'norm_mean' in data_dict else 0.0
        margin = intensities.shape - self.mask_size
        margin = np.maximum(margin, 0)
        
        intensities = intensities.copy()
        num_masks = self.rng.integers(self.max_masks, endpoint=True)
        for _ in range(num_masks):
            start_coords = self.rng.integers(margin, endpoint=True)
            slices = tuple([slice(l, l + dim) for l, dim in zip(start_coords, self.mask_size)])
            intensities[slices] = cval

        data_dict['intensities'] = intensities

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    mask_size={self.mask_size},',
            f'    max_masks={self.max_masks},',
            ')',
        ])
    

class FDIAsClass:

    def __init__(
        self,
        fdis: List[int]=[
            11, 12, 13, 14, 15, 16, 17, 18,
            21, 22, 23, 24, 25, 26, 27, 28,
            31, 32, 33, 34, 35, 36, 37, 38,
            41, 42, 43, 44, 45, 46, 47, 48,
        ],
        ignore_index: int=-100,
        **kwargs,
    ):
        fdis = np.array(fdis)
        inverse_fdis = np.full(fdis.max() + 1, -1)
        inverse_fdis[fdis] = np.arange(fdis.shape[0])

        self.fdis = fdis
        self.inverse_fdis = inverse_fdis
        self.ignore_index = ignore_index

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:        
        if (
            'labels' in data_dict
            and data_dict['labels'].size
            and data_dict['labels'].max() > 1
        ):
            # determine unique labels and reshape indices to 3D
            unique, inverse = np.unique(data_dict['labels'], return_inverse=True)
            inverse = inverse.reshape(data_dict['labels'].shape)

            # determine 3D masks of valid and invalid FDIs
            unique_is_valid = np.any(unique[None] == self.fdis[:, None], axis=0)
            is_valid = unique_is_valid[inverse]
            is_invalid = (unique[inverse] != 0) & ~is_valid

            # update valid FDIs to class index and invalid FDIs to ignore index
            data_dict['labels'][is_valid] = self.inverse_fdis[data_dict['labels'][is_valid]] + 1
            data_dict['labels'][is_invalid] = self.ignore_index
            
        if 'unique_labels' in data_dict:
            is_valid = np.any(data_dict['unique_labels'][None] == self.fdis[:, None], axis=0)
            data_dict['unique_labels'][is_valid] = self.inverse_fdis[data_dict['unique_labels'][is_valid]]
            data_dict['unique_labels'][~is_valid] = self.ignore_index

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    fdis={self.fdis},',
            f'    ignore_index={self.ignore_index},',
            ')',
        ])
    

class HistogramNormalize:

    def __init__(
        self,
        filter_size: int=25,
        min_peak_width: int=5,
        min_peak_height: float=0.0002,
        fit_peak_width: int=200,
        sigma_factor: float=2.33,
        **kwargs,
    ):
        self.filter_size = filter_size
        self.min_peak_width = min_peak_width
        self.min_peak_height = min_peak_height
        self.fit_peak_width = fit_peak_width
        self.sigma_factor = sigma_factor

    def gauss(self, x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def gauss_fit(self, x, y):
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(self.gauss, x, y, p0=[max(y), mean, sigma])
        
        return popt
    
    def determine_peak(
        self,
        intensities,
        **data_dict,
    ):
        # determine histogram of air or denser
        clipped = np.maximum(intensities, -1000)
        unique, counts = np.unique(clipped, return_counts=True)
        unique = unique.astype(int)
        hist = np.zeros(unique.max() - unique.min() + 1, int)
        hist[unique - unique.min()] = counts

        # fix histogram with all even or all odd intensities
        if np.all((intensities % 2) == (intensities[0, 0, 0] % 2)):
            hist[1::2] = hist[:-1:2]
            hist = hist // 2

        # determine top of soft tissue peak in histogram
        hist_filter = ndimage.median_filter(hist, size=self.filter_size)
        deriv = ndimage.gaussian_filter1d(
            input=hist_filter,
            sigma=20, order=1,
        )
        transitions = np.concatenate((
            [False],
            ~((deriv[1:] > 0) ^ (deriv[:-1] > 0))
        ))
        peaks = transitions * ndimage.minimum_filter1d(
            input=hist_filter >= self.min_peak_height * intensities.size,
            size=self.min_peak_width,
        )
        boundaries = np.nonzero(peaks[:-1] ^ peaks[1:])[0]
        peak_idx = boundaries[-4] + hist_filter[boundaries[-4]:].argmax()

        # determine head of peak around peak top
        x = np.arange(
            unique.min() + peak_idx - self.fit_peak_width,
            unique.min() + peak_idx + self.fit_peak_width + 1,
        )
        peak = hist_filter[peak_idx - self.fit_peak_width:peak_idx + self.fit_peak_width + 1]

        return x, peak

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # fit gaussian distribution around soft tissue peak
        x, peak = self.determine_peak(intensities, **data_dict)
        _, mean, sigma = self.gauss_fit(x, peak)

        # truncate scan information by clipping soft tissue and extreme values
        bone_value = mean + self.sigma_factor * sigma
        max_value = np.quantile(intensities, 0.995)
        clipped = intensities.clip(bone_value, max_value)

        data_dict['intensities'] = intensities
        data_dict['norm_min'] = bone_value.astype(intensities.dtype)
        data_dict['norm_max'] = max_value.astype(intensities.dtype)
        data_dict['norm_mean'] = clipped.mean().astype(intensities.dtype)
        data_dict['norm_std'] = clipped.std().astype(intensities.dtype)

        return data_dict
    
    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    filter_size={self.filter_size},',
            f'    min_peak_width={self.min_peak_width},',
            f'    min_peak_height={self.min_peak_height},',
            f'    fit_peak_width={self.fit_peak_width},',
            f'    sigma_factor={self.sigma_factor},',
            ')',
        ])


class IntensityAsFeatures:

    def __init__(
        self,
        norm_clip: Optional[Tuple[int, int]],
        norm_method: Literal['unit', 'symmetric', 'standard'],
    ):
        assert norm_method in ['unit', 'symmetric', 'standard']

        self.norm_clip = norm_clip
        self.norm_method = norm_method

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict['intensities'] = intensities

        # clip intensities to sensible range
        norm_clip = self.norm_clip if self.norm_clip is not None else (
            data_dict['norm_min'], data_dict['norm_max'],
        )
        features = intensities.clip(*norm_clip).astype(float)

        # normalize intensities to sensible location and scale
        if self.norm_method == 'unit':  # [0, 1]
            features = (features - norm_clip[0]) / (norm_clip[1] - norm_clip[0])
        elif self.norm_method == 'symmetric':  # [-1, 1]
            features = (features - norm_clip[0]) / (norm_clip[1] - norm_clip[0])
            features = 2 * features - 1
        elif self.norm_method == 'standard':  # N(0, 1)
            mean, std = data_dict['norm_mean'], data_dict['norm_std']
            features = (features - mean) / std

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], np.expand_dims(features, -4)),
            )
        else:
            data_dict['features'] = np.expand_dims(features, -4)

        return data_dict
    
    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    norm_clip={self.norm_clip},',
            f'    norm_method={self.norm_method},',
            ')',
        ])


class ToTensor:

    def __init__(
        self,
        bool_dtypes: List[np.dtype]=[bool, np.bool8],
        int_dtypes: List[np.dtype]=[int, np.uint8, np.int16, np.uint16, np.int32, np.int64],
        float_dtypes: List[np.dtype]=[float, np.float32, np.float64],
        str_dtypes: List[np.dtype]=[str, np.str_],
    ) -> None:
        self.bool_dtypes = bool_dtypes
        self.int_dtypes = int_dtypes
        self.float_dtypes = float_dtypes
        self.str_dtypes = str_dtypes

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, TensorType[..., Any]]:
        for k, v in data_dict.items():
            dtype = v.dtype if isinstance(v, np.ndarray) else type(v)
            if dtype in self.bool_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.bool)
            elif dtype in self.int_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.int64)
            elif dtype in self.float_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.float32)
            elif dtype in self.str_dtypes:
                continue
            else:
                raise ValueError(
                    'Expected a scalar or NumPy array with elements of '
                    f'{self.bool_dtypes + self.int_dtypes + self.float_dtypes},'
                    f' but got {dtype}.'
                )
            
        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
