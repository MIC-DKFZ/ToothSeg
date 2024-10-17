from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

from scipy import ndimage
import nibabel
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchtyping import TensorType

import baselines.data.transforms as T
from baselines.models.cluster import fast_search_cluster


class EncoderDecoderModule(pl.LightningModule):

    def __init__(
        self,
        lr: float=0.001,
        weight_decay: float=0.01,
        crop_size: Union[Tuple[int, int, int], int]=256,
        crop_stride: Union[Tuple[int, int, int], int]=128,
        patch_size: int=96,
        epochs: int=1000,
        use_scheduler: bool=True,
        sigma: float=1.0,
        return_type: Literal['fdi', 'iso', 'binary', 'instances']='fdi',
        out_dir: Union[List[Path], Path]=Path('output'),
        **kwargs,
    ):
        if isinstance(crop_size, int):
            crop_size = (crop_size,)*3
        if isinstance(crop_stride, int):
            crop_stride = (crop_stride,)*3
        if isinstance(patch_size, int):
            patch_size = (patch_size,)*3

        super().__init__()

        self.tooth_patches = T.ToothPatches(patch_size, sigma=sigma)

        self.lr = lr
        self.weight_decay = weight_decay
        self.crop_size = crop_size
        self.crop_stride = crop_stride
        self.patch_size = patch_size
        self.epochs = epochs
        self.use_scheduler = use_scheduler
        self.return_type = return_type
        self.out_dirs = [out_dir] if not isinstance(out_dir, list) else out_dir

    def crop_slices(
        self,
        features: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        factor: float=1.0,
    ) -> list[Tuple[slice, slice, slice, slice, slice]]:
        crop_size = tuple(map(lambda s: int(np.ceil(factor * s)), self.crop_size))
        crop_stride = tuple([int(np.ceil(factor * stride)) for stride in self.crop_stride])
        shape = tuple(map(lambda s: int(np.ceil(factor * s)), features.shape[-3:]))

        start_idxs = torch.cartesian_prod(*[
            torch.cat((
                torch.arange(0, dim - size, stride),
                torch.tensor([dim - size]),
            )) for dim, size, stride in zip(shape, crop_size, crop_stride)
        ])

        slice_idxs = torch.stack((
            start_idxs, start_idxs + torch.tensor(crop_size),
        ), dim=-1).tolist()
        slices = [
            (slice(None), slice(None)) + tuple([slice(l, r) for l, r in idxs])
            for idxs in slice_idxs
        ]

        return slices
    
    def gaussian_kernel(
        self,
        factor: float=1.0,
        sigma_factor: float=1/8,
    ) -> np.ndarray:
        crop_size = tuple(map(lambda s: int(factor * s), self.crop_size))

        tmp = np.zeros(crop_size)
        center_coords = [i // 2 for i in crop_size]
        sigmas = [i * sigma_factor for i in crop_size]
        tmp[tuple(center_coords)] = 1
        gaussian = ndimage.gaussian_filter(
            tmp, sigmas, 0, mode='constant', cval=0,
        )
        gaussian /= np.max(gaussian) * 1
        gaussian[gaussian == 0] = np.min(gaussian[gaussian != 0])

        return gaussian

    def cluster(
        self,
        seg: TensorType['B', 1, 'D', 'H', 'W', torch.bool],
        offsets: TensorType['B', 3, 'D', 'H', 'W', torch.float32],
    ) -> TensorType['N', torch.int64]:
        seg = seg[0, 0]
        offsets = offsets[0].permute(1, 2, 3, 0)

        pos_coords = torch.nonzero(seg)
        pred_voxels = pos_coords + offsets[seg]
    
        _, cluster_idxs = fast_search_cluster(pred_voxels)

        return cluster_idxs
    
    def tooth_representations(
        self,
        seg: TensorType['B', 1, 'D', 'H', 'W', torch.bool],
        offsets: TensorType['B', 3, 'D', 'H', 'W', torch.float32],
        cluster_idxs: TensorType['N', torch.int64],
        min_density: int=4,
    ) -> TensorType['D', 'H', 'W', torch.int64]:
        seg = seg[0, 0]
        offsets = offsets[0].permute(1, 2, 3, 0)

        pos_coords = torch.nonzero(seg)
        pred_voxels = (pos_coords + offsets[seg]).long()

        # filter voxels that are predicted too few times
        _, inverse, counts = torch.unique(pred_voxels, dim=0, return_inverse=True, return_counts=True)
        dense_mask = (counts >= min_density)[inverse]
        pred_voxels = torch.clip(
            pred_voxels[dense_mask],
            torch.tensor(0).to(pred_voxels),
            torch.tensor(seg.shape).to(pred_voxels) - 1,
        )
        cluster_idxs = cluster_idxs[dense_mask]

        out = torch.zeros_like(seg).long()
        if pred_voxels.shape[0] > 0:
            out[tuple(pred_voxels.T)] = cluster_idxs

        return out
    
    def tooth_instances(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        instances: TensorType['I', 'D', 'H', 'W', torch.int64],
    ) -> Tuple[
        TensorType['P', 'C+1', 'D', 'H', 'W', torch.float32],
        list[Tuple[slice, slice, slice, slice, slice]],
    ]:
        data_dict = self.tooth_patches(
            intensities=x[0, 0].cpu().numpy(),
            instances=instances.cpu().numpy(),
        )

        patches = np.concatenate(
            (data_dict['intensities'][:, None], data_dict['instances']), axis=1,
        )
        tooth_crops = torch.from_numpy(patches).to(x)

        patch_idxs = data_dict['patches_idxs']
        patch_slices = [
            tuple([slice(l, r) for l, r in idxs]) for idxs in patch_idxs
        ]

        return tooth_crops, patch_slices
    
    def pad(
        self,
        volume: TensorType['B', 'C', 'd', 'h', 'w', torch.int64],
        crop_size: Tuple[int, int, int],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W'],
        List[Tuple[int, int]],
    ]:
        pads = [max(c - d, 0) for c, d in zip(crop_size, volume.shape[-3:])]
        pads = [(pad // 2, pad // 2 + pad % 2) for pad in pads]
        
        torch_pads = [p for pad in pads[::-1] for p in pad]
        out = torch.nn.functional.pad(volume, torch_pads, value=volume.amin())
        
        return out, pads
    
    def resample(
        self,
        volume: TensorType['B', 'C', 'd', 'h', 'w', torch.int64],
        affine: TensorType[4, 4, torch.float32],
        shape: TensorType[3, torch.int64],
    ) -> TensorType['D', 'H', 'W', torch.int64]:
        volume = volume[0, 0].cpu().numpy()
        affine = affine.cpu().numpy()

        orientation = nibabel.io_orientation(np.linalg.inv(affine))
        affine_ori = affine @ nibabel.orientations.inv_ornt_aff(orientation, shape.cpu().numpy())
        offsets = affine_ori[:3, 3]

        # pad volume outside of ROI
        volume = np.pad(volume, [
            (max(0, int(np.ceil(-offset))), 0)
            for offset in offsets
        ])
        offsets = np.maximum(offsets, 0)

        # remove volume due to padding
        volume = volume[int(offsets[0]):, int(offsets[1]):, int(offsets[2]):]
        affine[:3, 3] = 0

        # rescale volume to original spacing
        spacing = 1 / np.linalg.norm(affine[:, :3], axis=0)
        if not np.allclose(spacing, 1.0):
            out_size = tuple((spacing * volume.shape).astype(int).tolist())
            volume = torch.from_numpy(volume.copy())

            max_seg = torch.full(out_size, 0.5)
            ret = torch.zeros(out_size, dtype=torch.int64)
            for label in torch.unique(volume)[1:]:
                label_out = torch.nn.functional.interpolate(
                    (volume[None, None] == label).float(), out_size, mode='trilinear',
                )[0, 0]
                ret = torch.where(max_seg < label_out, label, ret)
        else:
            ret = torch.from_numpy(volume)

        # insert ROI into original volumee
        assert np.all(orientation[:, 0] == np.arange(3))
        ret = ret[:shape[0], :shape[1], :shape[2]]
        seg = torch.zeros(shape.tolist(), dtype=int)
        seg[:ret.shape[0], :ret.shape[1], :ret.shape[2]] = ret
        
        # orient volume to original orientation
        seg = nibabel.apply_orientation(
            arr=seg, ornt=orientation,
        )

        return seg

    def load_ckpt(self, model: nn.Module, config: Dict[str, Any]):
        state_dict = model.state_dict()
        ckpt = config['pretrained']
        if ckpt:
            ckpt = torch.load(ckpt)['state_dict']
            ckpt = {k.split('.', 1)[1]: v for k, v in ckpt.items()}
            ckpt = {k: v for k, v in ckpt.items() if k in state_dict}
            model.load_state_dict(ckpt)
            model.requires_grad_(False)

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[torch.optim.lr_scheduler._LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if not self.use_scheduler:
            return opt

        sch = CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=1e-7,
        )

        return [opt], [sch]
    
    def save_output(
        self,
        instances: TensorType[1, 'C', 'D', 'H', 'W', torch.int64],
        scan_file: Path,
        affine: TensorType[4, 4, torch.float32],
        shape: TensorType[3, torch.int64],
    ):
        for i, out_dir in enumerate(self.out_dirs):
            out = self.resample(instances[:, i:i+1], affine, shape)

            if self.return_type == 'fdi':
                out[out > 0] = T.FDIAsClass().fdis[out[out > 0] - 1]

            pred = out.astype(np.uint16)
            
            scan_file = self.trainer.predict_dataloaders.dataset.root / scan_file
            scan_nii = nibabel.load(scan_file)

            out_dir = self.trainer.predict_dataloaders.dataset.root / out_dir
            out_dir.mkdir(exist_ok=True)

            img = nibabel.Nifti1Image(pred, scan_nii.affine)
            nibabel.save(img, out_dir / scan_file.name)
