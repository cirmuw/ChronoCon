
import pydicom
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterable
from monai.transforms import LoadImage
import pydicom
from tqdm import tqdm

def filter_image_paths(image_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Attempts to load each path with MONAI's LoadImage transform.
    Returns:
      valid_paths: all paths that loaded successfully
      error_paths: all paths that raised an exception
    """
    loader = LoadImage(image_only=True, ensure_channel_first=True)

    valid_paths = []
    error_paths = []

    for path in tqdm(image_paths, desc="Filtering image paths"):
        try:
            _ = loader(path)
            valid_paths.append(path)
        except Exception as e:
            print(f"Warning: Could not load {path}. Error: {e}")
            error_paths.append(path)

    return valid_paths, error_paths

# def extract_landmarks_from_df(dfm, image_idx=0, landmark_columns = None):
#     if not isinstance(landmark_columns, Iterable) and landmark_columns == None: 
#         landmark_columns = dfm.filter(regex="^landmark").columns
#     landmarks = dfm.loc[image_idx, landmark_columns].values.reshape(-1, 2)
#     return landmarks


def extract_landmarks_from_df(dfm, image_idx=0, landmark_names=None, x_suffix="-X", y_suffix="-Y"):
    """
    Extracts landmarks (x, y) from a DataFrame row.

    Parameters:
    - dfm: pandas DataFrame containing landmark columns.
    - image_idx: row index to extract landmarks from.
    - landmark_columns: list of landmark base names or column names (optional).
    - x_suffix: suffix for x-coordinates (default: "-X").
    - y_suffix: suffix for y-coordinates (default: "-Y").

    Returns:
    - landmarks: numpy array of shape (N, 2)
    """
    
    if landmark_names is None or not isinstance(landmark_names, Iterable):
        # Automatically select columns starting with 'landmark'
        landmark_names = dfm.filter(regex="^landmark").columns
        landmarks = dfm.loc[image_idx, landmark_names].values.reshape(-1, 2)
    
    else:
        # Assume landmark_columns are base names, and construct full column names
        x_cols = [f"{col}{x_suffix}" for col in landmark_names]
        y_cols = [f"{col}{y_suffix}" for col in landmark_names]

        # Check that all columns exist
        missing = set(x_cols + y_cols) - set(dfm.columns)
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
        
        # Extract values
        x_vals = dfm.loc[image_idx, x_cols].values
        y_vals = dfm.loc[image_idx, y_cols].values
        landmarks = np.stack((x_vals, y_vals), axis=1)

    return landmarks


# def extract_extras_from_filename(filename: str, ending = ".dcm", replace_ending=True, filename_str="filename"): 
#     x = filename.split(ending)[0].split("_")
#     d = {
#         filename_str: filename.replace(ending,"") if replace_ending else filename, 
#         "id": x[0],
#         "date_str": x[1],
#         "region": x[2],
#         "left_or_right": x[3],
#         # I dont know what the rest means:  dp_MTwo_InvNo_RotNo_BOk_OPNo_app_ComNo
#         "x4": x[4],
#         "x5": x[5],
#         "x6": x[6],
#         "x7": x[7],
#         "x8": x[8],     
#         "x9": x[9],    
#         "x10": x[10],    
#         "x11": x[11]                                        
#     }
#     return d


from pathlib import Path

def extract_extras_from_filename(
        filename: str,
        ending=".npy",
        replace_ending=False,
        filename_str="file_name"
):
    # strip the file suffix you expect
    stem = Path(filename).name
    if ending and stem.endswith(ending):
        stem = stem[: -len(ending)]
    parts = stem.split("_")

    keys = [
        "id", "date_str", "region", "left_or_right",
        "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"
    ]

    record = {filename_str: filename if not replace_ending else stem}
    # fill missing parts with None so we SEE them
    for k, key in enumerate(keys):
        record[key] = parts[k] if k < len(parts) else None
    return record

def extract_extras_from_abspath(abs_path):
    filename = str(abs_path).replace("._","").split(str("/"))[-1]
    return {**{"image": abs_path, **extract_extras_from_filename(filename)}}




def get_dicom_info(dicom_paths, only_header=False):
    """
    Given a list of DICOM file paths, returns a DataFrame where each row 
    corresponds to one file, indexed by the file path. 
    The columns include:
      - dim_original              : Shape of the pixel array
      - pixel_spacing            : Pixel spacing if present in the DICOM metadata
      - intensity_range_original : (min, max) of the pixel data
      - intensity_005            : 0.05% intensity quantile
      - intensity_500            : 50% intensity quantile (median)
      - intensity_995            : 99.5% intensity quantile
    """
    records = []

    for path in dicom_paths:
        try:
            ds = pydicom.dcmread(path)
            pixel_spacing = getattr(ds, "PixelSpacing", None)
            if only_header:
                records.append({
                    "file_path": path,
                    "pixel_spacing": pixel_spacing,
                    "pixel_spacing_0": pixel_spacing[0] if pixel_spacing != None else None,
                    "pixel_spacing_1": pixel_spacing[1] if pixel_spacing != None else None
                })
            if not only_header:
                pixel_array = ds.pixel_array
                dim_original = pixel_array.shape
                intensity_range_original = (pixel_array.min(), pixel_array.max())
                q_005, q_500, q_995 = np.quantile(pixel_array, [0.0005, 0.5, 0.995])
                    
                records.append({
                    "file_path": path,
                    "dim_original": dim_original,
                    "pixel_spacing": pixel_spacing,
                    "pixel_spacing_0": pixel_spacing[0] if pixel_spacing != None else None,
                    "pixel_spacing_1": pixel_spacing[1] if pixel_spacing != None else None,
                    "intensity_min": intensity_range_original[0],
                    "intensity_005": q_005,
                    "intensity_500": q_500,
                    "intensity_995": q_995,
                    "intensity_max": intensity_range_original[1]
                })
        except Exception as e:
            # In case a file is not a valid DICOM or any other read error occurs
            print(f"Warning: Could not process {path}. Error: {e}")
            continue

    # Convert to DataFrame and set the index to the file path
    df = pd.DataFrame(records).set_index("file_path")
    return df