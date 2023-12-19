### 1. Convert Raw Data from dcm to nii.gz
````
python process_raw.py
````

- **Gather and Filter DICOM Files:**
    Collect all .dcm files. Excluding specific files based on incorrect labels or geometry issues.
    Reference ignore_files.txt for a list of files to be omitted.
- **Conversion to NIfTI Format:**
    Convert the collected .dcm images and labels into .nii.gz format.
- **Label Conversion:**
    Transform labels from ISO numbering format (https://en.wikipedia.org/wiki/Dental_notation) to a continuous ascending format.
- **Incorporate Updates:**
    For certain images, class labels may have been corrected or segmentations adjusted. The updated got integrate into the processed database.
- **Final Dataset Structure:**
    The end result will be organized in the Dataset164_All_Classes/ folder.
  - imagesTr subfolder: Contains all images.
  - labelsTr subfolder: Contains all labels.
- **Class Mapping:**
    Class mapping from the Raw (ISO) format to the Dataset164_All_Classes format:
  - Background -> 0
  - Lower Jaw -> 1
  - Upper Jaw -> 2
  - Dental Implant -> 3
  - Non-tooth Supported Crown -> 4
  - [11 - 18] -> [5 - 12]
  - [21 - 28] -> [13 - 20]
  - [31 - 38] -> [21 - 28]
  - [41 - 48] -> [29 - 36]
  - [51 - 55] -> [37 - 41]
  - [61 - 65] -> [42 - 46]
  - [71 - 75] -> [47 - 51]
  - [81 - 85] -> [52 - 56]


### 2. Filter Labels to only contain Permanent Teeth
````
python process_filtered_classes.py
````
- **Generate Filtered Dataset:** Utilize the Dataset164_All_Classes/ as a starting point to produce Dataset164_Filtered_Classes/.

- **Focus on Permanent Teeth Classes:** Dataset164_Filtered_Classes/ includes only Permanent Teeth classes.

- **Exclude Primary Teeth Data:** Ignore any files containing Primary Teeth from Dataset164_Filtered_Classes/.

- **Class Mapping for Filtered Dataset:**
    Class mapping from the Raw (ISO) format to the Dataset164_Filtered_Classes format:
  - background -> 0
  - [11 - 18] -> [1 - 8]
  - [21 - 28] -> [9 - 16]
  - [31 - 38] -> [17 - 24]
  - [41 - 48] -> [25 - 32]