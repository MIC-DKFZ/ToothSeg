from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import open3d
from scipy import ndimage
import nibabel

import baselines.data.transforms as T


def wolla(path, threshold: int, min_voxels: int, dilate: int=0, color: str='double'):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)

    transform = T.HistogramNormalize()
    data_dict = transform(intensities=data)


    affine = img.affine
    threshold = data_dict['norm_mean'] + 2.5 * data_dict['norm_std']
    mask = data >= threshold

    if dilate:
        mask = ndimage.binary_dilation(
            input=mask,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilate,
        )

    labels, _ = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    _, inverse, counts = np.unique(labels.flatten(), return_inverse=True, return_counts=True)

    labels[(counts < min_voxels)[inverse].reshape(labels.shape)] = 0

    points = np.column_stack(labels.nonzero()).astype(float)
    hom_points = np.column_stack((points, np.ones_like(points[:, 0])))
    points = np.einsum('ij,kj->ki', affine, hom_points)
    points = points[:, :-1]

    if color == 'double':
        colors = points[:, 0] - points[:, 0].mean()
        colors[colors > 0] = (colors[colors > 0] - colors[colors > 0].min()) / (colors[colors > 0].max() - colors[colors > 0].min())
        colors[colors < 0] = 1 - (colors[colors < 0] - colors[colors < 0].min()) / (colors[colors < 0].max() - colors[colors < 0].min())
    elif color == 'single':
        colors = points[:, 0] - points[:, 0].min()
        colors = (colors - colors.min()) / (colors.max() - colors.min())

    # colors = np.abs(colors - colors.mean())
    # colors = (colors - colors.min()) / (colors.max() - colors.min())
    colors = np.tile(colors, (3, 1)).T

    return labels, points, colors


def visualize(points: NDArray[np.float32], colors: NDArray[np.int64]) -> None:
    """Try to initialize window to visualize provided point cloud."""
    # intialize Open3D point cloud
    pcd = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(points),
    )
    pcd.colors = open3d.utility.Vector3dVector(colors)

    # initialize Open3D window
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    # add each provided geometry to the Open3D window
    vis.add_geometry(pcd)

    # point size
    options = vis.get_render_option()
    options.point_size = 2

    # camera options
    view = vis.get_view_control()
    view.change_field_of_view(-55.0)  # defaut 60 - 55 = minimum 5
    view.set_zoom(0.66)
    view.set_front(np.array([[0., 1., 0.]]).T)
    view.set_up(np.array([[0., 0., 1.]]).T)
    
    # actually render the scene of geometries
    vis.run()


if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/tooth_segmentation/data')

    scan_dir = root / 'Dataset164_Filtered_Classes/imagesTr'

    for scan_path in scan_dir.glob('*'):
        labels, points, colors = wolla(scan_path, 400, 10_000, 1)
        visualize(points, colors)