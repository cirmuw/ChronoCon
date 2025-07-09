import napari
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from ra_utils.annotation.point_annotator import point_annotator

# === CONFIGURATION ===
def main():
    # 👇 Replace this with your actual DICOM path
    DICOM_PATH = "/home/clemens/data/tmp_dicom/RA/1.2.826.0.1.3680043.8.760.0.1281924843.1539345414455.27703/1.2.826.0.1.3680043.8.760.0.1281924843.1539345414455.27704/1.2.826.0.1.3680043.8.760.0.1281924843.1539345414455.27705"
    

    LANDMARK_NAMES = ['TMT1-D',
                        'MTB1',
                        'MTP1-P',
                        'MTP1-D',
                        'FIP1-P',
                        'FIP1-D',
                        'FDP1-P',
                        'TMT2-D',
                        'MTB2',
                        'MTP2-P',
                        'MTP2-D',
                        'FIP2-P',
                        'TMT3-D',
                        'MTB3',
                        'MTP3-P',
                        'MTP3-D',
                        'FIP3-P',
                        'TMT4-D',
                        'MTB4',
                        'MTP4-P',
                        'MTP4-D',
                        'FIP4-P',
                        'TMT5-D',
                        'MTB5',
                        'MTP5-P',
                        'MTP5-D',
                        'FIP5-P']
    
    LANDMARK_NAMES_with_numbers = [ f"{i} {name}" for i, name in enumerate(LANDMARK_NAMES) ]
    
    point_annotator(DICOM_PATH, labels=LANDMARK_NAMES_with_numbers)

# im_path = '<path to>/openfield-Pranav-2018-10-30/labeled-data/m4s1/*.png'
#point_annotator(im_path, labels=['ear_l', 'ear_r', 'tail'])

if __name__ == "__main__":
    main()

    


