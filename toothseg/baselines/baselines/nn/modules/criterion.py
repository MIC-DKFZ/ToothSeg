from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from baselines.nn.modules.loss import lovasz_hinge


class BinarySegmentationLoss(nn.Module):
    "Implements binary segmentation loss function."

    def __init__(
        self,
        bce_weight: float=1.0,
        dice_weight: float=0.0,
        focal_weight: float=0.0,
        focal_alpha: float=0.25,
        focal_gamma: float=2.0,
    ) -> None:
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def forward(
        self,
        pred: TensorType['B', 1, '...', torch.float32],
        target: TensorType['B', '...', torch.int64],
    ) -> TensorType[torch.float32]:
        pred = pred.squeeze(1)
        target = target.float()
        
        loss = self.bce_weight * self.bce(pred, target)

        if self.dice_weight:        
            probs = torch.sigmoid(pred)
            dim = tuple(range(1, len(probs.shape)))
            numerator = 2 * torch.sum(probs * target, dim=dim)
            denominator = torch.sum(probs ** 2, dim=dim) + torch.sum(target ** 2, dim=dim)
            dice_loss = 1 - torch.mean((numerator + 1) / (denominator + 1))

            loss += self.dice_weight * dice_loss

        if self.focal_weight:
            bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

            loss += self.focal_weight * focal_loss.mean()

        return loss
    

class SegmentationLoss(nn.Module):

    def __init__(
        self,
        ce_weight: float=1.0,
        dice_weight: float=1.0,
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def __call__(
        self,
        pred: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        target: TensorType['B', 'D', 'H', 'W', torch.int64],
    ) -> TensorType[torch.float32]:
        ce_loss = self.ce(pred, target)

        probs = torch.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.shape[1])
        target = target.permute(0, 4, 1, 2, 3)

        dim = tuple(range(2, len(probs.shape)))
        numerator = 2 * torch.sum(probs * target, dim=dim)
        denominator = torch.sum(probs ** 2, dim=dim) + torch.sum(target ** 2, dim=dim)
        dice_loss = 1 - torch.mean((numerator + 1) / (denominator + 1))

        loss = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
        )

        return loss


class SpatialEmbeddingLoss(nn.Module):

    def __init__(
        self,
        crop_size: Tuple[int, int, int],
        voxel_size: float,
        learn_center: bool=True,
        learn_ellipsoid: bool=True,
        w_foreground: float=1.0,
        w_instance: float=1.0,
        w_smooth: float=1.0,
        w_seed: float=10.0,
    ):
        super().__init__()

        self.learn_center = learn_center
        self.n_sigma = 3 if learn_ellipsoid else 1

        self.w_foreground = w_foreground
        self.w_instance = w_instance
        self.w_smooth = w_smooth
        self.w_seed = w_seed

        # coordinate map
        xm = torch.linspace(0, crop_size[2] * voxel_size, crop_size[2]).view(
            1, 1, -1).expand(*crop_size)
        ym = torch.linspace(0, crop_size[1] * voxel_size, crop_size[1]).view(
            1, -1, 1).expand(*crop_size)
        zm = torch.linspace(0, crop_size[0] * voxel_size, crop_size[0]).view(
            -1, 1, 1).expand(*crop_size)
        xyzm = torch.stack((xm, ym, zm), 0)

        self.register_buffer('xyzm', xyzm)

    def forward(
        self,
        pred_offsets: TensorType['B', 3, 'D', 'H', 'W', torch.float32],
        pred_sigmas: TensorType['B', '1|3', 'D', 'H', 'W', torch.float32],
        pred_seeds: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
        targets: TensorType['B', 'D', 'H', 'W', torch.int64],
    ):
        """
        'k' index represents instance
        'i' index represents voxel
        """

        batch_size, _, depth, height, width = pred_offsets.shape

        loss = 0

        for b in range(0, batch_size):

            spatial_emb = torch.tanh(pred_offsets[b]) + self.xyzm  # 3 x d x h x w
            sigma = pred_sigmas[b]  # 1|3 x d x h x w
            seed_map = torch.sigmoid(pred_seeds[b])  # 1 x d x h x w

            # loss accumulators
            smooth_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instances = targets[b].unsqueeze(0)  # 1 x d x h x w

            # regress bg to zero
            bg_mask = instances == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))

            for k in instances.unique()[1:]:
                mask_k = instances == k  # 1 x d x h x w

                # predict center of attraction (\hat{C}_k)
                if self.learn_center:
                    center_k = spatial_emb[mask_k.expand_as(spatial_emb)].view(
                        3, -1).mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1 x 1
                else:
                    center_k = self.xyzm[mask_k.expand_as(self.xyzm)].view(
                        3, -1).mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1 x 1

                # calculate sigma
                sigmas_ki = sigma[mask_k.expand_as(
                    sigma)].view(self.n_sigma, -1)
                sigma_k = sigmas_ki.mean(1).view(  # (\hat{\sigma}_k)
                    self.n_sigma, 1, 1, 1)  # 1|3 x 1 x 1 x 1

                # calculate smooth loss before exp
                smooth_loss = smooth_loss + torch.mean(
                    torch.pow(sigmas_ki - sigma_k[..., 0, 0].detach(), 2),
                )

                # exponential to effectively predict 1 / (2 * sigma_k**2)
                sigma_k = torch.exp(sigma_k * 10)

                # calculate gaussian
                probs_i = torch.exp(-1 * torch.sum(
                    sigma_k * torch.pow(spatial_emb - center_k, 2),
                    dim=0,
                    keepdim=True,
                ))  # 1 x d x h x w

                # apply lovasz-hinge loss
                logits_i = 2 * probs_i - 1
                instance_loss = instance_loss + lovasz_hinge(logits_i, mask_k)

                # seed loss
                seed_loss += self.w_foreground * torch.sum(
                    torch.pow(seed_map[mask_k] - probs_i[mask_k].detach(), 2),
                )

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                smooth_loss /= obj_count

            seed_loss = seed_loss / (depth * height * width)

            loss += (
                self.w_instance * instance_loss
                + self.w_smooth * smooth_loss
                + self.w_seed * seed_loss
            )

        loss = loss / batch_size

        return loss + pred_offsets.sum()*0


class IdentificationLoss(nn.Module):

    def __init__(
        self,
        w_ce: float=1.0,
        w_focal: float=1.0,
        w_homo: float=1.0,
        alpha: float=0.25,
        gamma: float=2.0,
    ):
        super().__init__()

        self.w_ce = w_ce
        self.w_focal = w_focal
        self.w_homo = w_homo
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        prototypes: List[TensorType['F', 'N', torch.float32]],
        classes: TensorType['K', 'C', torch.float32],
        targets: TensorType['B', 'D', 'H', 'W', torch.int64],
    ):
        keep_idxs = torch.nonzero(torch.all(~torch.isnan(classes), dim=1))[:, 0]
        if keep_idxs.shape[0] == 0:
            return classes.sum()*0

        prototypes = [prototypes[idx] for idx in keep_idxs]
        classes = classes[keep_idxs]
        target = torch.cat([target.unique()[1:] - 1 for target in targets])[keep_idxs]

        ce_loss = F.cross_entropy(classes, target, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        homo_loss = torch.zeros_like(ce_loss[:0])
        for proto in prototypes:
            homo = torch.mean(proto.T - proto.mean(1))
            homo_loss = torch.cat((homo_loss, homo[None]))

        loss = (
            self.w_ce * ce_loss.mean()
            + self.w_focal * focal_loss.mean()
            + self.w_homo * homo_loss.mean()
        )

        return loss
