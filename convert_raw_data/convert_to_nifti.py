from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import pydicom
from batchgenerators.utilities.file_and_folder_operations import *

from convert_raw_data.label_handling import convert_shanks_labels_to_mine

def psg_to_npa(path):
    """
    THIS CODE IF FROM SHANK, NOT US!

    PSG format version 2
    To convert a segmentation file to numpy array (2d or 3d)
    Later version might support 4d as well
    Note that for interchange between this code and backend, 3D x and y need to be swapped

    :param path: complete path inc. file (str)
    :return: npa (np array), object_type label (str), object_id label (str)
    """

    header_chunksizes = [1, 2, 2, 2]
    header_values = []

    with open(path, "rb") as f:
        for header_chunksize in header_chunksizes:
            chunk = f.read(header_chunksize)
            i = int.from_bytes(chunk, byteorder="big")
            header_values.append(i)
        for i in range(2):
            string_size = int.from_bytes(f.read(1), byteorder="big")
            str_encoded = f.read(string_size)
            string = str_encoded.decode("utf-8")
            header_values.append(string)

        version_is_supported = True
        if not header_values[0] == 2:
            print(
                "psg_to_npa: incorrect load function for file version, unsupported: {}".format(
                    header_values[0]
                )
            )
            version_is_supported = False
        assert version_is_supported

        if header_values[3] == 0:
            npa = np.zeros((header_values[1] * header_values[2]), dtype=bool)
        else:
            npa = np.zeros(
                (header_values[1] * header_values[2] * header_values[3]), dtype=bool
            )

        while True:
            chunk_start = f.read(4)
            chunk_end = f.read(4)
            if chunk_start:
                index_start = int.from_bytes(chunk_start, byteorder="big")
            else:
                break
            if chunk_end:
                index_end = int.from_bytes(chunk_end, byteorder="big")
                npa[index_start:index_end] = True
            else:
                npa[index_start:] = True
                break

    if header_values[3] == 0:
        npa = np.reshape(npa, (header_values[1], header_values[2]))
        npa = np.swapaxes(npa, 0, 1)
    else:
        npa = np.reshape(npa, (header_values[1], header_values[2], header_values[3]))
        npa = np.swapaxes(npa, 0, 1)

    object_type = header_values[4]
    object_id = header_values[5]

    # Needed since for some cases in the updated data the header information differs from the file name
    # If this is the Case use the file name as object id
    name_value=path.rsplit("/",1)[1].split("_")[1].replace(".psg","")
    if object_type=="TOOTH" and object_id!=name_value:
        print(f"WARNING: header value {object_id} differs from file name {name_value}.psg -> Label {name_value} is used")
        object_id=name_value
    return npa, object_type, object_id


def convert_case(folder: str):
    try:
        # we use this order to determine what is allowed to overwrite what
        subfolder_order = [
            'LOWER_JAW',
            'UPPER_JAW',
            'TOOTH',
            'SUPERNUMERARY_TOOTH',
            'PRIMARY_TOOTH',
            'NON_TOOTH_SUPPORTED_CROWN',
            'DENTAL_IMPLANT',
            'METAL_FILLING',
            'METAL_CROWN',
        ]

        scan_folder = join(folder, 'scan')
        dcm_files = subfiles(scan_folder, suffix='.dcm')
        assert len(dcm_files) == 1, f'Scan file not found for patient {folder}'
        ds = pydicom.read_file(dcm_files[0])
        if hasattr(ds, 'PixelSpacing'):
            in_plane = [float(i) for i in ds.PixelSpacing]
        else:
            in_plane = None
        if hasattr(ds, 'SliceThickness'):
            slice_thickness = float(ds.SliceThickness)
        else:
            slice_thickness = None
        if in_plane is not None and slice_thickness is not None:
            spacing = [slice_thickness] + in_plane
            image = ds.pixel_array
        else:
            print(folder, 'fallback to itk')
            img = sitk.ReadImage(dcm_files[0])
            spacing = list(img.GetSpacing())[::-1]
            image = sitk.GetArrayFromImage(img)

        seg = np.zeros(image.shape)
        psg_folder = join(folder, 'psg_manual_ann')
        psg_subfolders = subfolders(psg_folder, join=False)
        remaining_subfolders = [i for i in subfolder_order if i in psg_subfolders]
        # now append additional folders
        remaining_subfolders += [i for i in psg_subfolders if i not in remaining_subfolders]

        for psg_fld in remaining_subfolders:
            psg_files = subfiles(join(psg_folder, psg_fld), suffix='.psg')
            for psg_file in psg_files:
                seg_here, label_text, label_int = psg_to_npa(psg_file)
                label_here = convert_shanks_labels_to_mine(label_text, label_int)
                # for real?
                seg_here = seg_here.transpose((2, 0, 1))[::-1]
                # we ignore overwriting jaws.
                if np.sum(seg[seg_here != 0] > 2) > 100:
                    overwritten = np.unique(seg[seg_here != 0])
                    print(folder)
                    print(folder, '\noverwriting labels', overwritten, '\n', 'num_pixels', np.sum(seg[seg_here != 0]), '\n',
                          'with label', label_here)
                seg[seg_here != 0] = label_here

        image_itk = sitk.GetImageFromArray(image)
        image_itk.SetSpacing(spacing[::-1])
        sitk.WriteImage(image_itk, join(folder, 'image.nii.gz'))
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk.SetSpacing(spacing[::-1])
        sitk.WriteImage(seg_itk, join(folder, 'seg.nii.gz'))
        print(folder, image.shape, list(spacing), '\n')
    except Exception as e:
        print(e, '\n', folder)


if __name__ == '__main__':

    p = Pool(16)
    #base = '/home/isensee/drives/E132-Projekte/Projects/2022_Isensee_Shank_cbCT_teeth/raw_data'
    base = "/home/l727r/Downloads/output/new_labels"
    cases = subfolders(base)
    r = p.map_async(convert_case, cases)
    p.close()
    p.join()
    # tmp = '/home/fabian/temp/shank_data/carley-primitive-cephalopod'
    # convert_case(tmp)

