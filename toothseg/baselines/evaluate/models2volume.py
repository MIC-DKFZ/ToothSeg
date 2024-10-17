from pathlib import Path
import re
from typing import List

import nibabel
import numpy as np
import open3d
import pandas as pd
import pymeshlab
from tqdm import tqdm



def get_fdi_map():
    fdis = np.array([
        0, 11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
    ])

    fdi_map = np.zeros(max(fdis) + 1)
    fdi_map[fdis] = np.arange(len(fdis))

    return fdi_map.astype(int)



def process_patient(
    scan_dir: Path,
    meshes_dir: Path,
    out_dir: Path,
    skip: List[str],
):    
    scan_nii = nibabel.load(list(sorted(scan_dir.glob('imaging*')))[-1])
    scan = np.asarray(scan_nii.dataobj)

    labels = np.zeros_like(scan).astype(np.uint8)
    ms = pymeshlab.MeshSet()
    if 'Segmentation' in meshes_dir.as_posix():
        mesh_paths = list(meshes_dir.glob('*'))
    else:
        mesh_paths = list(meshes_dir.glob('tooth*')) + list(meshes_dir.glob('Tooth*'))
    for mesh_path in tqdm(mesh_paths):
        if 'Segmentation' in mesh_path.as_posix():
            fdi = mesh_path.as_posix().split('_')[-2]
        else:
            fdi = mesh_path.stem[6:8]
        if fdi in skip:
            continue

        ms.load_new_mesh(str(mesh_path))

        vertices = ms.current_mesh().vertex_matrix()
        triangles = ms.current_mesh().face_matrix()

        vertices_hom = np.column_stack((
            vertices, np.ones_like(vertices[:, :1]),
        ))

        vertices = (vertices_hom @ np.linalg.inv(scan_nii.affine).T)[:, :-1]

        slices = ()
        for i in range(3):
            start, stop = np.floor(vertices[:, i].min()), np.ceil(vertices[:, i].max())
            slc = slice(int(start), int(stop))
            slices = slices + (slc,)

        grid = np.stack(np.meshgrid(
            *[np.arange(slc.start, slc.stop) for slc in slices],
            indexing='ij',
        ), axis=-1)
        query_points = grid.reshape(-1, 3) + 0.5


        mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
        scene = open3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
        
        query_points = open3d.core.Tensor(query_points, dtype=open3d.core.Dtype.Float32)
        occupancy = scene.compute_occupancy(query_points)
        occupancy = occupancy.numpy().reshape(grid.shape[:-1])

        mask = np.zeros_like(scan, dtype=bool)
        mask[slices] = occupancy > 0
        if re.match('\d\d', fdi):
            label = get_fdi_map()[int(fdi)]
        else:
            label = 33
        labels[mask] = label

    scan = scan[:, :, ::-1]
    scan_nii = nibabel.Nifti1Image(scan.astype(np.int16), scan_nii.affine)
    nibabel.save(scan_nii, out_dir.parent / 'cbcts' / f'{scan_dir.name}.nii.gz')    

    labels = labels[::-1, ::-1, ::-1]
    labels_nii = nibabel.Nifti1Image(labels.astype(np.uint16), scan_nii.affine)

    out_dir.mkdir(exist_ok=True)
    nibabel.save(labels_nii, out_dir / f'{scan_dir.name}.nii.gz')
    



if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/tooth_segmentation/commercial')

    for system in ['Segmentation_teeth_lower_upper_jaw', 'RELU_Shank', 'Diagno_DVTMA5_fixed']:
        df = pd.read_excel(root / system / f'map_{system}.xlsx', dtype={'skip': str})

        for i, row in df.iterrows():
            print(row['case1'])
            skip = row['skip'].split(',') if isinstance(row['skip'], str) else []
            print(skip)
            process_patient(
                root / 'Test_Data' / row['case2'],
                root / system / row['case1'],
                root / 'Test_Data' / system,
                skip,
            )
