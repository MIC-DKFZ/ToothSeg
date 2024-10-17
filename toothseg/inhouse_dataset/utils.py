import SimpleITK as sitk
import numpy as np


def psg_to_npa(path):
    """
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
            npa = np.zeros((header_values[1] * header_values[2] * header_values[3]), dtype=bool)

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
    name_value = path.rsplit("/", 1)[1].split("_")[1].replace(".psg", "")
    if object_type == "TOOTH" and object_id != name_value:
        print(
            f"WARNING: header value {object_id} differs from file name {name_value}.psg -> Label"
            f" {name_value} is used"
        )
        object_id = name_value
    return npa, object_type, object_id


def copy_geometry(image: sitk.Image, ref: sitk.Image):
    """
    copy geometry from ref to image
    :param image:
    :param ref:
    :return:
    """
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


def convert_ISO_to_continious(img):
    dct = {10 + i: 4 + i for i in range(1, 9)}  # [11 - 18] -> [5 - 12]
    dct.update({20 + i: 12 + i for i in range(1, 9)})  # [21 - 28] -> [13 - 20]
    dct.update({30 + i: 20 + i for i in range(1, 9)})  # [31 - 38] -> [21 - 28]
    dct.update({40 + i: 28 + i for i in range(1, 9)})  # [41 - 48] -> [29 - 36]

    # in addition some kids got milk teeth. Oof. up to 5 more per row
    dct.update({50 + i: 36 + i for i in range(1, 6)})
    dct.update({60 + i: 41 + i for i in range(1, 6)})
    dct.update({70 + i: 46 + i for i in range(1, 6)})
    dct.update({80 + i: 51 + i for i in range(1, 6)})

    new_labels = np.zeros(img.shape, dtype=np.uint8)
    for val in np.unique(img):
        if val == 0:
            continue
        new_val = dct[int(val)]
        x, y, z = np.where(img == val)
        new_labels[x, y, z] = new_val
    return new_labels


def convert_label_ISO_to_continious(label_text: str, label_int: str):
    if label_text == "LOWER_JAW":
        label_int = 1
    elif label_text == "UPPER_JAW":
        label_int = 2
    elif label_text == "DENTAL_IMPLANT":
        label_int = 3
    elif label_text == "NON_TOOTH_SUPPORTED_CROWN":
        label_int = 4
    else:
        # up to 8 teeth per row. oof
        assert label_int != ""
        dct = {10 + i: 4 + i for i in range(1, 9)}  # [11 - 18] -> [5 - 12]
        dct.update({20 + i: 12 + i for i in range(1, 9)})  # [21 - 28] -> [13 - 20]
        dct.update({30 + i: 20 + i for i in range(1, 9)})  # [31 - 38] -> [21 - 28]
        dct.update({40 + i: 28 + i for i in range(1, 9)})  # [41 - 48] -> [29 - 36]

        # in addition some kids got milk teeth. Oof. up to 5 more per row
        dct.update({50 + i: 36 + i for i in range(1, 6)})
        dct.update({60 + i: 41 + i for i in range(1, 6)})
        dct.update({70 + i: 46 + i for i in range(1, 6)})
        dct.update({80 + i: 51 + i for i in range(1, 6)})

        label_int = dct[int(label_int)]
    return label_int
