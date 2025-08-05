from pathlib import Path
import pandas as pd
import numpy as np
import json


import pydicom
from pydicom.errors import InvalidDicomError
import re
import ra_utils
import ra_utils.data.data_utils
from ra_utils.data.data_utils import (
    extract_extras_from_filename, 
    extract_extras_from_abspath
)

import yaml
import json
import os

def read_dict_from_file(filename: str) -> dict:
    """
    Reads a YAML (.yml/.yaml) or JSON (.json) file and returns its contents as a dictionary.
    
    Parameters:
    - filename (str): Path to the YAML or JSON file.

    Returns:
    - dict: Dictionary containing the file contents.

    Raises:
    - ValueError: If file extension is not .yml/.yaml/.json.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    if ext in ['.yml', '.yaml']:
        with open(filename, 'r') as file:
            return yaml.safe_load(file)
    elif ext == '.json':
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Please use a .yml, .yaml, or .json file.")


class DataHandler_CR_autoscoRA(object):
    def __init__(self, 
                 folder_H_images = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_dicoms",
                 folder_F_images = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms",
                 df_lm_labels_H = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/100_all_H_joints36/points.csv",
                 df_lm_labels_F = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/100_all_F_joints27/points.csv",
                 df_autoscoRA_labels_F = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_data/autoscoRA_feet.csv",
                 df_autoscoRA_labels_H = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_data/autoscoRA_hands.csv",
                 training_test_splits_json_H = None,
                 training_test_splits_json_F = None,
                 df_autoscoRA_labels_F_header = "infer",
                 df_autoscoRA_labels_H_header = "infer",
                 ):
        
        self.filepaths = dict(
                 folder_H_images = Path(folder_H_images),
                 folder_F_images = Path(folder_F_images),
                 df_lm_labels_H = Path(df_lm_labels_H),
                 df_lm_labels_F = Path(df_lm_labels_F),
                 df_autoscoRA_labels_F = Path(df_autoscoRA_labels_F),
                 df_autoscoRA_labels_H = Path(df_autoscoRA_labels_H),
                 training_test_splits_json_F=training_test_splits_json_F,
                 training_test_splits_json_H=training_test_splits_json_H
        )
        self.df_autoscoRA_labels_F_header = df_autoscoRA_labels_F_header
        self.df_autoscoRA_labels_H_header = df_autoscoRA_labels_H_header
        self.load_everything()

    @staticmethod
    def load_landmark_df(landmark_path: Path| str, header="infer"):
        df = pd.read_csv(landmark_path, header=header)
        df = df.rename(columns={"img": "filename"})
        if header == None: 
            column_names = ["filename"] + [f"landmark_{(i-1) // 2}_{(i-1) % 2}" 
                                        for i in range(1, df.shape[1])]
            df.columns = column_names
        
        # Drop rows where all landmark columns are zero
        m = ~(df.iloc[:,1:] == 0).all(axis=1)
        df = df[m]
        return df

    def load_H_and_F_landmark_df(self):
        df_lm_labels_H = self.load_landmark_df(self.filepaths["df_lm_labels_H"], header=self.df_autoscoRA_labels_H_header)
        df_lm_labels_F = self.load_landmark_df(self.filepaths["df_lm_labels_F"], header=self.df_autoscoRA_labels_F_header)
        return df_lm_labels_H, df_lm_labels_F

    def load_H_and_F_paths_from_folder(self):
        files_H = list(self.filepaths["folder_H_images"].glob("*.dcm"))
        files_F = list(self.filepaths["folder_F_images"].glob("*.dcm"))
        
        # ignore the files starting with "._"
        files_H = [f for f in files_H if not f.name.startswith("._")]
        files_F = [f for f in files_F if not f.name.startswith("._")]
        
        files_H_with_extras = [extract_extras_from_abspath(file, ending=".dcm", replace_ending=True) for file in files_H]
        files_F_with_extras = [extract_extras_from_abspath(file, ending=".dcm", replace_ending=True) for file in files_F]
        
        df_images_H = pd.DataFrame(files_H_with_extras).rename(columns={"file_name": "filename"})
        df_images_F = pd.DataFrame(files_F_with_extras).rename(columns={"file_name": "filename"})
        return df_images_H, df_images_F

    def load_autoscoRA_lables(self):
        df_autoscoRA_labels_H = pd.read_csv(self.filepaths["df_autoscoRA_labels_H"])
        df_autoscoRA_labels_F = pd.read_csv(self.filepaths["df_autoscoRA_labels_F"])
        return df_autoscoRA_labels_H, df_autoscoRA_labels_F

    def load_everything(self):
        self.df_lm_labels_H, self.df_lm_labels_F = self.load_H_and_F_landmark_df()
        
        self.landmark_names_H = [re.sub("((-X)|(-Y)|(_0)|_1)$", '', i) for i in self.df_lm_labels_H.columns[1:]][::2] if self.df_autoscoRA_labels_H_header == "infer" else None          
        self.landmark_names_F = [re.sub("((-X)|(-Y)|(_0)|_1)$", '', i) for i in self.df_lm_labels_F.columns[1:]][::2] if self.df_autoscoRA_labels_F_header == "infer" else None          
        
        #self.landmark_names_H = list(set([re.sub("((-X)|(-Y)|(_0)|_1)$", '', i) for i in self.df_lm_labels_H.columns[1:]])) if self.df_autoscoRA_labels_H_header == "infer" else None          
        #self.landmark_names_F = list(set([re.sub("((-X)|(-Y)|(_0)|_1)$", '', i) for i in self.df_lm_labels_F.columns[1:]])) if self.df_autoscoRA_labels_F_header == "infer" else None          
        self.df_images_H, self.df_images_F       = self.load_H_and_F_paths_from_folder()
        self.df_autoscoRA_labels_H, self.df_autoscoRA_labels_F = self.load_autoscoRA_lables()
        
        self.df_images_and_landmarks_F = pd.merge(
            self.df_images_F, 
            self.df_lm_labels_F, 
            on="filename", 
            how="inner"
        )
        self.df_images_and_landmarks_H = pd.merge(
            self.df_images_H, 
            self.df_lm_labels_H, 
            on="filename", 
            how="inner"
        )

        # Load splits if provided
        p = self.filepaths["training_test_splits_json_H"]
        if p is not None:
            data = read_dict_from_file(p)
            self.splits_dict_H_raw = data
            # Potentially exclude certain IDs
            # Example: ids_to_exclude = data["exclude"]
            self.splits_dict_H = self.splits_dict_H_raw.copy()
        else: 
            self.splits_dict_H = None

        p = self.filepaths["training_test_splits_json_F"]
        if p is not None:
            data = read_dict_from_file(p)
            self.splits_dict_F_raw = data
            self.splits_dict_F = self.splits_dict_F_raw.copy()
        else: 
            self.splits_dict_F = None

    # --------------------------------------------------------------
    # Helper method: Extract image paths and landmark arrays from df
    # --------------------------------------------------------------
    def _extract_image_paths_and_landmarks(self, df_subset: pd.DataFrame, get_pixel_spacing=False, landmark_names=None, pixel_spacing_default = [0.1, 0.1]):
        """
        Given a subset of self.df_images_and_landmarks_H (or F),
        return a list of image-path strings, a 3D numpy array of landmarks,
        and optionally a 2D numpy array of pixel spacings.
        """
        # Convert the 'image' column (Path objects) to string paths
        image_paths = df_subset["image"].apply(str).to_list()

        # Build landmarks array (example snippet)
        landmarks_list = [
            ra_utils.data.data_utils.extract_landmarks_from_df(df_subset, image_idx=i, landmark_names=landmark_names)
            for i, _row in df_subset.iterrows()
        ]
        landmarks_array = np.array(landmarks_list, dtype=np.uint16)
        landmarks_array = np.flip(landmarks_array, axis=-1)  # (x,y)->(y,x) if you want that

        # Build pixel_spacing array if requested
        pixel_spacing_array = None
        if get_pixel_spacing:
            pixel_spacings = []
            for idx, row in df_subset.iterrows():
                dcm_path = row["image"]
                try:
                    # Force read, so pydicom tries to parse even if 'DICM' header is missing
                    ds = pydicom.dcmread(str(dcm_path), force=True)

                    # Attempt to read PixelSpacing
                    try:
                        ps = ds.PixelSpacing  # typically [RowSpacing, ColSpacing]
                    except (AttributeError, KeyError):
                        print(f"[WARNING] PixelSpacing missing in {dcm_path}, using {pixel_spacing_default}.")
                        ps = pixel_spacing_default

                except InvalidDicomError:
                    # If pydicom cannot parse the file at all
                    print(f"[WARNING] {dcm_path} is not a valid DICOM. Using {pixel_spacing_default}.")
                    ps = pixel_spacing_default

                pixel_spacings.append(ps)

            # Convert to numpy: shape (N, 2)
            pixel_spacing_array = np.array(pixel_spacings, dtype=float)

        return image_paths, landmarks_array, pixel_spacing_array



    # --------------------------------------------------------------
    #  get_landmarks_dataset_H
    # --------------------------------------------------------------
    def get_landmarks_dataset_H(self, get_pixel_spacing=False, pixel_spacing_default = [0.1, 0.1]):
        """
        Returns the dataset splits for the 'hands' data as a tuple:
            (image_paths_train,   image_paths_test1,   image_paths_test2,
             landmarks_train,     landmarks_test1,     landmarks_test2,
             pixel_spacing_train, pixel_spacing_test1, pixel_spacing_test2)
        """
        if self.splits_dict_H is None:
            raise ValueError("No splits_dict_H loaded! Provide training_test_splits_json_H.")

        if not all(k in self.splits_dict_H for k in ["training", "test1", "test2"]):
            raise ValueError("splits_dict_H is missing required keys: 'training', 'test1', 'test2'")

        # Filenames from JSON
        train_filenames = self.splits_dict_H["training"]
        test1_filenames = self.splits_dict_H["test1"]
        test2_filenames = self.splits_dict_H["test2"]

        if train_filenames == None: 
            train_filenames = []
        if test1_filenames == None:
            test1_filenames = []
        if test2_filenames == None:
            test2_filenames = []
        

        # Filter DataFrame
        df_train = self.df_images_and_landmarks_H[
            self.df_images_and_landmarks_H["filename"].isin(train_filenames)
        ]
        df_test1 = self.df_images_and_landmarks_H[
            self.df_images_and_landmarks_H["filename"].isin(test1_filenames)
        ]
        df_test2 = self.df_images_and_landmarks_H[
            self.df_images_and_landmarks_H["filename"].isin(test2_filenames)
        ]

        # Extract for each subset
        landmark_names = self.landmark_names_H
        (image_paths_train, landmarks_train, pixel_spacing_train) = \
            self._extract_image_paths_and_landmarks(df_train, get_pixel_spacing=get_pixel_spacing, landmark_names=landmark_names, pixel_spacing_default = pixel_spacing_default)

        (image_paths_test1, landmarks_test1, pixel_spacing_test1) = \
            self._extract_image_paths_and_landmarks(df_test1, get_pixel_spacing=get_pixel_spacing, landmark_names=landmark_names, pixel_spacing_default = pixel_spacing_default)

        (image_paths_test2, landmarks_test2, pixel_spacing_test2) = \
            self._extract_image_paths_and_landmarks(df_test2, get_pixel_spacing=get_pixel_spacing, landmark_names=landmark_names, pixel_spacing_default = pixel_spacing_default)

        return (
            image_paths_train, 
            image_paths_test1,
            image_paths_test2,
            landmarks_train,
            landmarks_test1,
            landmarks_test2,
            pixel_spacing_train,
            pixel_spacing_test1,
            pixel_spacing_test2
        )

    # --------------------------------------------------------------
    #  get_landmarks_dataset_H_CV
    # --------------------------------------------------------------
    def get_landmarks_dataset_H_CV(self):
        """
        Returns cross-validation folds for 'hands' data.
        Each fold is a tuple of the form:
            (train_image_paths, train_landmarks,
             test_image_paths,  test_landmarks)
        """
        if self.splits_dict_H is None:
            raise ValueError("No splits_dict_H loaded! Provide training_test_splits_json_H.")
        
        if "cv" not in self.splits_dict_H:
            raise ValueError("splits_dict_H has no 'cv' key for cross-validation folds.")
        
        folds_data = []
        for fold_idx, fold in enumerate(self.splits_dict_H["cv"]):
            if not all(k in fold for k in ["train", "test"]):
                raise ValueError(f"Fold {fold_idx} missing 'train' or 'test' keys.")
            
            train_filenames = fold["train"]
            test_filenames  = fold["test"]

            df_train = self.df_images_and_landmarks_H[
                self.df_images_and_landmarks_H["filename"].isin(train_filenames)
            ]
            df_test  = self.df_images_and_landmarks_H[
                self.df_images_and_landmarks_H["filename"].isin(test_filenames)
            ]

            landmark_names = self.landmark_names_H
            (train_image_paths, train_landmarks, _) = \
                self._extract_image_paths_and_landmarks(df_train, get_pixel_spacing=False, landmark_names=landmark_names)
            (test_image_paths,  test_landmarks,  _) = \
                self._extract_image_paths_and_landmarks(df_test, get_pixel_spacing=False, landmark_names=landmark_names)

            folds_data.append((train_image_paths, train_landmarks,
                               test_image_paths,  test_landmarks))

        return folds_data

    # --------------------------------------------------------------
    #  get_landmarks_dataset_F
    # --------------------------------------------------------------
    def get_landmarks_dataset_F(self, get_pixel_spacing=False, pixel_spacing_default = [0.1, 0.1]):
        """
        Returns the dataset splits for the 'feet' data as a tuple:
            (image_paths_train,   image_paths_test1,   image_paths_test2,
             landmarks_train,     landmarks_test1,     landmarks_test2,
             pixel_spacing_train, pixel_spacing_test1, pixel_spacing_test2)
        """
        if self.splits_dict_F is None:
            raise ValueError("No splits_dict_F loaded! Provide training_test_splits_json_F.")

        if not all(k in self.splits_dict_F for k in ["training", "test1", "test2"]):
            raise ValueError("splits_dict_F is missing required keys: 'training', 'test1', 'test2'")

        train_filenames = self.splits_dict_F["training"]
        test1_filenames = self.splits_dict_F["test1"]
        test2_filenames = self.splits_dict_F["test2"]

        df_train = self.df_images_and_landmarks_F[
            self.df_images_and_landmarks_F["filename"].isin(train_filenames)
        ]
        df_test1 = self.df_images_and_landmarks_F[
            self.df_images_and_landmarks_F["filename"].isin(test1_filenames)
        ]
        df_test2 = self.df_images_and_landmarks_F[
            self.df_images_and_landmarks_F["filename"].isin(test2_filenames)
        ]

        landmark_names = self.landmark_names_F
        (image_paths_train, landmarks_train, pixel_spacing_train) = \
            self._extract_image_paths_and_landmarks(df_train, get_pixel_spacing=get_pixel_spacing, landmark_names=landmark_names, pixel_spacing_default = pixel_spacing_default)

        (image_paths_test1, landmarks_test1, pixel_spacing_test1) = \
            self._extract_image_paths_and_landmarks(df_test1, get_pixel_spacing=get_pixel_spacing, landmark_names=landmark_names, pixel_spacing_default = pixel_spacing_default)

        (image_paths_test2, landmarks_test2, pixel_spacing_test2) = \
            self._extract_image_paths_and_landmarks(df_test2, get_pixel_spacing=get_pixel_spacing, landmark_names=landmark_names, pixel_spacing_default = pixel_spacing_default)

        return (
            image_paths_train, 
            image_paths_test1,
            image_paths_test2,
            landmarks_train,
            landmarks_test1,
            landmarks_test2,
            pixel_spacing_train,
            pixel_spacing_test1,
            pixel_spacing_test2
        )

    # --------------------------------------------------------------
    #  get_landmarks_dataset_F_CV
    # --------------------------------------------------------------
    def get_landmarks_dataset_F_CV(self):
        """
        Returns cross-validation folds for 'feet' data.
        Each fold is a tuple:
            (train_image_paths, train_landmarks,
             test_image_paths,  test_landmarks)
        """
        if self.splits_dict_F is None:
            raise ValueError("No splits_dict_F loaded! Provide training_test_splits_json_F.")
        
        if "cv" not in self.splits_dict_F:
            raise ValueError("splits_dict_F has no 'cv' key for cross-validation folds.")

        folds_data = []
        for fold_idx, fold in enumerate(self.splits_dict_F["cv"]):
            if not all(k in fold for k in ["train", "test"]):
                raise ValueError(f"Fold {fold_idx} missing 'train' or 'test' keys.")

            train_filenames = fold["train"]
            test_filenames  = fold["test"]

            df_train = self.df_images_and_landmarks_F[
                self.df_images_and_landmarks_F["filename"].isin(train_filenames)
            ]
            df_test  = self.df_images_and_landmarks_F[
                self.df_images_and_landmarks_F["filename"].isin(test_filenames)
            ]

            landmark_names = self.landmark_names_F
            (train_image_paths, train_landmarks, _) = \
                self._extract_image_paths_and_landmarks(df_train, get_pixel_spacing=False, landmark_names=landmark_names)
            (test_image_paths,  test_landmarks,  _) = \
                self._extract_image_paths_and_landmarks(df_test, get_pixel_spacing=False, landmark_names=landmark_names)

            folds_data.append((train_image_paths, train_landmarks,
                               test_image_paths,  test_landmarks))

        return folds_data
