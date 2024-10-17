from pathlib import Path

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch
from tqdm import tqdm


if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/tooth_segmentation/data/Dataset164_Filtered_Classes/imagesTr/')

    ckpt = torch.load('/home/mkaailab/Documents/logs/cuinet/cuinet_single_tooth_9/checkpoints/weights-epoch=259.ckpt')
    ckpt2 = torch.load('checkpoints/cuinet_skeletons.ckpt')

    ckpt2['state_dict']['model.block_five_a.conv.0.weight'] = ckpt2['state_dict'].pop('model.block_five.conv.0.weight')
    ckpt2['state_dict']['model.block_five_a.conv.0.bias'] = ckpt2['state_dict'].pop('model.block_five.conv.0.bias')
    ckpt2['state_dict']['model.block_five_a.conv.1.weight'] = ckpt2['state_dict'].pop('model.block_five.conv.1.weight')
    ckpt2['state_dict']['model.block_five_a.conv.1.bias'] = ckpt2['state_dict'].pop('model.block_five.conv.1.bias')
    ckpt2['state_dict']['model.block_five_a.conv.3.weight'] = ckpt2['state_dict'].pop('model.block_five.conv.3.weight')
    ckpt2['state_dict']['model.block_five_a.conv.3.bias'] = ckpt2['state_dict'].pop('model.block_five.conv.3.bias')
    ckpt2['state_dict']['model.block_five_a.conv.4.weight'] = ckpt2['state_dict'].pop('model.block_five.conv.4.weight')
    ckpt2['state_dict']['model.block_five_a.conv.4.bias'] = ckpt2['state_dict'].pop('model.block_five.conv.4.bias')
    ckpt2['state_dict']['model.block_five_b.conv.0.weight'] = ckpt2['state_dict'].pop('model.block_five.conv.6.weight')
    ckpt2['state_dict']['model.block_five_b.conv.0.bias'] = ckpt2['state_dict'].pop('model.block_five.conv.6.bias')
    ckpt2['state_dict']['model.block_five_b.conv.1.weight'] = ckpt2['state_dict'].pop('model.block_five.conv.7.weight')
    ckpt2['state_dict']['model.block_five_b.conv.1.bias'] = ckpt2['state_dict'].pop('model.block_five.conv.7.bias')

    torch.save(ckpt2, 'checkpoints/cuinet_skeletons_fixed.ckpt')

    img_paths, all_dims = [], []
    for img_path in tqdm(list(root.glob('*.nii.gz'))):
        img_nii = nibabel.load(root / img_path)

        shape = np.array(img_nii.header.get_data_shape())
        spacing = np.array(img_nii.header.get_zooms())

        if np.any(spacing > 0.3):
            continue

        dims = shape * spacing
        all_dims.append(dims)

        img_paths.append(img_path)

    all_dims = np.stack(all_dims)

    for i in np.argsort(all_dims.sum(-1)):
        print(img_paths[i])

        # image = np.asarray(img_nii.dataobj)
        # image = np.clip(image, 0, 2500)

        # plt.imshow(image.max(1))
        # plt.show()
