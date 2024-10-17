from typing import Literal

import numpy as np
import torch
from torchtyping import TensorType


def min_distance(distances, densities):
    n_points = distances.shape[0]
    sort_rho_idx = torch.argsort(densities, descending=True, stable=True)
    
    delta = torch.zeros(n_points).to(distances)
    nneigh = torch.zeros(n_points).to(densities)
    for i in range(1, n_points):
        old_i = sort_rho_idx[i]
        old_js = sort_rho_idx[torch.arange(i)]
        delta[old_i] = distances[old_i, old_js].amin()
        nneigh[old_i] = sort_rho_idx[distances[old_i, old_js].argmin()]
    delta[sort_rho_idx[0]] = distances[sort_rho_idx[0]].max()

    return delta, nneigh


def fast_search_cluster(
    coords,
    density_threshold: int=20,
    distance_threshold: int=10,
    method: Literal['distance', 'neighbor']='distance',
):
    coords_ = torch.asarray(coords)

    if coords_.shape[0] == 0:
        if isinstance(coords, np.ndarray):
            return coords[:0].astype(int), coords[:0, :0].astype(int)
        return coords[:0].long(), coords[:0, :0].long()

    voxel_idxs = coords_.detach().long()
    unique_coords, inverse, densities = torch.unique(
        voxel_idxs, return_inverse=True, return_counts=True, dim=0,
    )
    try:
        distances = torch.linalg.norm(
            (unique_coords[None] - unique_coords[:, None]).float(), dim=-1,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise e
        
        distances = torch.linalg.norm(
            (unique_coords[None].short() - unique_coords[:, None].short()).half(), dim=-1,
        )

    
    delta, nneigh = min_distance(distances, densities)

    peak_mask = (densities >= density_threshold) & (delta >= distance_threshold)
    peak_idxs = torch.nonzero(peak_mask)[:, 0]
    if not torch.any(peak_mask):
        peak_idxs = densities.argmax(keepdims=True)
        peak_mask[peak_idxs[0]] = True
    peak_voxels = unique_coords[peak_mask].to(coords_).long()

    if method == 'distance':
        distances = torch.linalg.norm(
            (coords_.detach()[None] - peak_voxels[:, None]).float(), dim=-1,
        )
        cluster_idxs = distances.argmin(dim=0) + 1
    elif method == 'neighbor':
        cluster_idxs = torch.full(delta.shape, -1)
        cluster_idxs[peak_idxs] = torch.tensor(list(range(peak_idxs.shape[0])))
        ordrho = torch.argsort(densities, descending=True, stable=True)
        for i in range(delta.shape[0]):
            if cluster_idxs[ordrho[i]] == -1:
                cluster_idxs[ordrho[i]] = cluster_idxs[nneigh[ordrho[i]]]

        cluster_idxs = cluster_idxs[inverse]
    else:
        raise ValueError(f'Method not recognized: {method}')

    if isinstance(coords, np.ndarray):
        peak_voxels = peak_voxels.numpy()
        cluster_idxs = cluster_idxs.numpy()

    return peak_voxels, cluster_idxs


def learned_region_cluster(
    offsets: TensorType['B', 3, 'D', 'H', 'W', torch.float32],
    sigmas: TensorType['B', '1|3', 'D', 'H', 'W', torch.float32],
    seeds: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    voxel_size: float,
    min_seed_score: float=0.5,
    min_cluster_size: int=40,
    min_unclustered: float=0.5,
) -> TensorType['B', 'D', 'H', 'W', torch.int64]:
    # determine spatial embeddings
    depth, height, width = offsets.shape[-3:]
    xm = torch.linspace(0, width * voxel_size, width).view(
        1, 1, -1).expand(*seeds.shape[2:])
    ym = torch.linspace(0, height * voxel_size, height).view(
        1, -1, 1).expand(*seeds.shape[2:])
    zm = torch.linspace(0, depth * voxel_size, depth).view(
        -1, 1, 1).expand(*seeds.shape[2:])
    xyzm = torch.stack((xm, ym, zm), 0).to(seeds)
    n_sigma = sigmas.shape[1]

    out = torch.zeros_like(seeds).long()
    for b in range(offsets.shape[0]):
        # prepare inputs
        offsets_b = torch.tanh(offsets[b])
        sigmas_b = torch.exp(sigmas[b] * 10)
        seeds_b = torch.sigmoid(seeds[b])
        mask_b = seeds_b >= 0.5

        # determine instances by clustering
        spatial_embeds = offsets_b + xyzm
        instance_idx = 0
        while mask_b.sum() >= min_cluster_size:
            voxel_idx = (seeds_b * mask_b).argmax()
            mask_b[np.unravel_index(voxel_idx.cpu().numpy(), mask_b.shape)] = False
            if seeds_b.reshape(-1)[voxel_idx] < min_seed_score:
                break

            center = spatial_embeds.reshape(3, -1)[:, voxel_idx].reshape(3, 1, 1, 1)
            bandwidth = sigmas_b.reshape(n_sigma, -1)[:, voxel_idx].reshape(n_sigma, 1, 1, 1)
            probs = (seeds_b >= 0.5) * torch.exp(-1 * torch.sum(
                bandwidth * torch.pow(spatial_embeds - center, 2),
                dim=0,
                keepdim=True,
            ))  # 1 x d x h x w
            proposal = probs > 0.5

            num_voxels = proposal.sum()
            overlap = mask_b[proposal].sum() / num_voxels
            if num_voxels >= min_cluster_size and overlap >= min_unclustered:
                out[b, proposal] = instance_idx + 1
                instance_idx += 1
            
            mask_b[proposal] = False

    return out[:, 0]
