import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fdi_pair_distrs', required=False, type=Path,
        default='toothseg/datasets/inhouse_dataset/test_fdi_pair_distrs.json',
        help='Path to JSON file with Gaussian distributions modeling centroid differences of FDI pairs.',
    )
    args = parser.parse_args()
    
    with open(args.fdi_pair_distrs, 'r') as f:
        distrs = json.load(f)

    means = np.array(distrs['means'])
    covs = np.array(distrs['covs'])

    fig, axs = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(args.fdi_pair_distrs.as_posix())
    for i, name in zip(range(3),['left-right', 'anterior-posterior', 'inferior-superior']):
        # fig.suptitle(name)
        im = axs[0, i].imshow(means[:16, :16, i])
        axs[0, i].set_title(f'Upper arch {name} differences')
        axs[0, i].set_xticks(range(16), labels=list(map(str, range(1, 17))))
        axs[0, i].set_yticks(range(16), labels=list(map(str, range(1, 17))))
        clb = fig.colorbar(im)
        clb.ax.set_title('mm')
        im = axs[1, i].imshow(means[16:, 16:, i])
        axs[1, i].set_title(f'Lower arch {name} differences')
        axs[1, i].set_xticks(range(16), labels=list(map(str, range(17, 33))))
        axs[1, i].set_yticks(range(16), labels=list(map(str, range(17, 33))))
        clb = fig.colorbar(im)
        clb.ax.set_title('mm')
    plt.tight_layout()
    plt.show(block=True)
