from collections import defaultdict
from itertools import chain
from pathlib import Path
import multiprocessing as mp
from datetime import datetime

import nibabel
import numpy as np
import pydicom
from tqdm import tqdm


def get_fdis(nii_path):
    nii = nibabel.load(nii_path)
    labels = np.asarray(nii.dataobj)

    unique_labels = np.unique(labels.flatten())
    return unique_labels


def get_age(nii_path):
    stem = nii_path.name[:-7]
    dcm_path = dcm_root / stem / 'scan' / f'{stem}.dcm'
    dcm = pydicom.dcmread(dcm_path)

    if not hasattr(dcm, 'PatientBirthDate') or not dcm.PatientBirthDate:
        return None
    
    if not hasattr(dcm, 'StudyDate') or not dcm.StudyDate:
        return None

    try:
        birth_dt = datetime.strptime(dcm.PatientBirthDate, '%Y%m%d')
    except ValueError:
        birth_dt = datetime.strptime(dcm.PatientBirthDate, '%Y/%m/%d')

    try:
        scan_dt = datetime.strptime(dcm.StudyDate, '%Y%m%d')
    except ValueError:
        scan_dt = datetime.strptime(dcm.StudyDate, '%Y/%m/%d')

    age = scan_dt.year - birth_dt.year
    if birth_dt.month > scan_dt.month or (
        birth_dt.month == scan_dt.month and birth_dt.day > scan_dt.day
    ):
        age -= 1

    if age < 10:
        print(nii_path)


    return age


def get_sex(nii_path):
    stem = nii_path.name[:-7]
    dcm_path = dcm_root / stem / 'scan' / f'{stem}.dcm'
    dcm = pydicom.dcmread(dcm_path)

    if not hasattr(dcm, 'PatientSex') or not dcm.PatientSex:
        return None

    return dcm.PatientSex


def get_age_sex(nii_path):
    stem = nii_path.name[:-7]
    dcm_path = dcm_root / stem / 'scan' / f'{stem}.dcm'
    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    if not hasattr(dcm, 'PatientSex') or not dcm.PatientSex:
        sex = None
    else:
        sex = dcm.PatientSex


    if not hasattr(dcm, 'PatientBirthDate') or not dcm.PatientBirthDate:
        return None, sex
    
    if not hasattr(dcm, 'StudyDate') or not dcm.StudyDate:
        return None, sex

    try:
        birth_dt = datetime.strptime(dcm.PatientBirthDate, '%Y%m%d')
    except ValueError:
        try:
            birth_dt = datetime.strptime(dcm.PatientBirthDate, '%Y/%m/%d')
        except ValueError:
            return None, sex

    try:
        scan_dt = datetime.strptime(dcm.StudyDate, '%Y%m%d')
    except ValueError:
        try:
            scan_dt = datetime.strptime(dcm.StudyDate, '%Y/%m/%d')
        except ValueError:
            return None, sex

    age = scan_dt.year - birth_dt.year
    if birth_dt.month > scan_dt.month or (
        birth_dt.month == scan_dt.month and birth_dt.day > scan_dt.day
    ):
        age -= 1

    if age < 10:
        print(nii_path)

    return age, sex


def get_patient(nii_path):
    stem = nii_path.name[:-7]
    dcm_path = dcm_root / stem / 'scan' / f'{stem}.dcm'
    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    if not hasattr(dcm, 'PatientName'):
        return None
    
    return str(dcm.PatientName)


def get_manufacturer_device(nii_path):
    stem = nii_path.name[:-7]
    dcm_path = dcm_root / stem / 'scan' / f'{stem}.dcm'
    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    if not hasattr(dcm, 'Manufacturer') or not dcm.Manufacturer:
        return None, None
    
    if not hasattr(dcm, 'ManufacturerModelName') or not dcm.ManufacturerModelName:
        return None, None
    
    return str(dcm.Manufacturer), str(dcm.ManufacturerModelName)


def get_res_kvp_ma(nii_path):
    stem = nii_path.name[:-7]
    dcm_path = dcm_root / stem / 'scan' / f'{stem}.dcm'
    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    thick = getattr(dcm, 'SliceThickness', None)
    spacing = getattr(dcm, 'PixelSpacing', [None])[0]
    kvp = getattr(dcm, 'KVP', None)
    current = getattr(dcm, 'XRayTubeCurrent', None)
    
    return thick, spacing, kvp, current


def get_fov(nii_path):
    stem = nii_path.name[:-7]
    dcm_path = dcm_root / stem / 'scan' / f'{stem}.dcm'
    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    thick = getattr(dcm, 'SliceThickness', None)
    spacing = getattr(dcm, 'PixelSpacing', [None])[0]

    if spacing is None or thick is None:
        return None, None
    
    slices = getattr(dcm, 'NumberOfFrames', None)
    rows, cols = getattr(dcm, 'Rows', None), getattr(dcm, 'Columns', None)

    if rows is None:
        rows = cols
    if cols is None:
        cols = rows
    if rows is None and cols is None:
        return None, None
    if slices is None:
        return None, None
    
    height = thick * slices
    diameter = max(rows, cols) * spacing

    if height > 200:
        print(nii_path)

    return height, diameter    


fdis = [
        0,
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
    ]

if __name__ == '__main__':
    nii_roots = [
        Path('/mnt/diag/CBCT/tooth_segmentation/data/Dataset164_Filtered_Classes/labelsTr'),
        Path('/mnt/diag/CBCT/tooth_segmentation/data/Dataset164_Filtered_Classes/labelsTs')
    ]
    nii_paths = sorted(chain(*[root.glob('*.nii.gz') for root in nii_roots]))
    # nii_paths = nii_paths[:50]

    dcm_root = Path('/mnt/diag/CBCT/tooth_segmentation/data/Fabian_v0.1')



    heights, diameters = [], []
    with mp.Pool(64) as p:
        i = p.imap_unordered(get_fov, nii_paths)
        t = tqdm(i, total=len(nii_paths))
        for height, diameter in t:
            heights.append(height)
            diameters.append(diameter)
    out = np.column_stack((heights, diameters))
    np.savetxt('fovs.txt', out, fmt='%s;')

    thicks, spacings, kvps, currents = [], [], [], []
    with mp.Pool(1) as p:
        i = p.imap_unordered(get_res_kvp_ma, nii_paths)
        t = tqdm(i, total=len(nii_paths))
        for thick, spacing, kvp, current in t:
            thicks.append(thick)
            spacings.append(spacing)
            kvps.append(kvp)
            currents.append(current)
    out = np.column_stack((thicks, spacings, kvps, currents))
    np.savetxt('res_kvp_ma.txt', out, fmt='%s;')

    ages, sexes = [], []
    with mp.Pool(64) as p:
        i = p.imap_unordered(get_age_sex, nii_paths)
        t = tqdm(i, total=len(nii_paths))
        for age, sex in t:
            ages.append(age)
            sexes.append(sex)
    np.savetxt('ages.txt', np.array(ages), fmt='%s')
    np.savetxt('sexes.txt', np.array(sexes), fmt='%s')





    counts = defaultdict(int)
    with mp.Pool(64) as p:
        i = p.imap_unordered(get_fdis, nii_paths)
        t = tqdm(i, total=len(nii_paths))
        for unique in t:
            for label in unique:
                counts[label] += 1

    for label in sorted(counts):
        print(fdis[label], '&', counts[label], f'({int(100 * counts[label] / len(nii_paths))})')




