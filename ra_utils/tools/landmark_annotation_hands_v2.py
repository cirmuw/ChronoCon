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
    
    # LANDMARK_NAMES = [
    #     "0",
    #     "1",
    #     "2",
    #     "3",
    #     "4",
    #     "5",
    #     "6",
    #     "7"
    # ]    
    

    LANDMARK_NAMES = ['WRC',
                    'CMC1-D',
                    'MCB1',
                    'MCP1-P',
                    'MCP1-D',
                    'PIP1-P',
                    'PIP1-D',
                    'DIP1-P',
                    'CMC2-D',
                    'MCB2',
                    'MCP2-P',
                    'MCP2-D',
                    'PIP2-P',
                    'PIP2-D',
                    'DIP2-P',
                    'CMC3-D',
                    'MCB3',
                    'MCP3-P',
                    'MCP3-D',
                    'PIP3-P',
                    'PIP3-D',
                    'DIP3-P',
                    'CMC4-D',
                    'MCB4',
                    'MCP4-P',
                    'MCP4-D',
                    'PIP4-P',
                    'PIP4-D',
                    'DIP4-P',
                    'CMC5-D',
                    'MCB5',
                    'MCP5-P',
                    'MCP5-D',
                    'PIP5-P',
                    'PIP5-D',
                    'DIP5-P'] 
    
    LANDMARK_NAMES_with_numbers = [ f"{i} {name}" for i, name in enumerate(LANDMARK_NAMES) ]
    
    point_annotator(DICOM_PATH, labels=LANDMARK_NAMES_with_numbers)

# im_path = '<path to>/openfield-Pranav-2018-10-30/labeled-data/m4s1/*.png'
#point_annotator(im_path, labels=['ear_l', 'ear_r', 'tail'])

if __name__ == "__main__":
    main()

    


