import napari
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from ra_utils.annotation.point_annotator import point_annotator

# === CONFIGURATION ===
def main():
    # 👇 Replace this with your actual DICOM path
    DICOM_PATH = "/home/clemens/data/tmp_dicom/RA/1.2.826.0.1.3680043.8.760.0.1104948461.1538485112979.40839/1.2.826.0.1.3680043.8.760.1.1104948461.1538485112979.40842/1.2.826.0.1.3680043.8.760.2.1104948461.1538485112979.40843"

    # 👇 Define the landmark names in the expected click order
    
    LANDMARK_NAMES = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7"
    ]    
    
    point_annotator(DICOM_PATH, labels=LANDMARK_NAMES)

# im_path = '<path to>/openfield-Pranav-2018-10-30/labeled-data/m4s1/*.png'
#point_annotator(im_path, labels=['ear_l', 'ear_r', 'tail'])

if __name__ == "__main__":
    main()

    


