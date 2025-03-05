from pathlib import Path
import pandas as pd
import numpy as np
import json

import ra_utils
import ra_utils.data.data_utils
from  ra_utils.data.data_utils import (
    extract_extras_from_filename, 
    extract_extras_from_abspath
)




class DataHandler_CR_autoscoRA(object):
    def __init__(self, 
                 folder_H_images = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_dicoms",
                 folder_F_images = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms",
                 df_lm_labels_H = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/100_all_H_joints36/points.csv",
                 df_lm_labels_F = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/100_all_F_joints27/points.csv",
                 df_autoscoRA_labels_F = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_data/autoscoRA_feet.csv",
                 df_autoscoRA_labels_H = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_data/autoscoRA_hands.csv",
                 training_test_splits_json_H = None, #base_dir / "projects__autoscora/landmark_data/splits/splits_H_TD_25-03-05.json",
                 training_test_splits_json_F = None #base_dir / "projects__autoscora/landmark_data/splits/splits_F_TD_25-03-05.json"
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
        self.load_everything()
        
        
     

    @staticmethod
    def load_landmark_df(landmark_path: Path| str):
        df = pd.read_csv(landmark_path, header=None)
        column_names = ["filename"] + [f"landmark_{(i-1) // 2}_{(i-1) % 2}" for i in range(1, df.shape[1])]
        df.columns = column_names
        # drop those where on data is available
        m = ~(df.filter(regex="^landmark")==0).all(axis=1)
        df = df[m]
        return df


    def load_H_and_F_landmark_df(self):
        df_lm_labels_H = self.load_landmark_df(self.filepaths["df_lm_labels_H"])
        df_lm_labels_F = self.load_landmark_df(self.filepaths["df_lm_labels_F"])
        return df_lm_labels_H, df_lm_labels_F

    def load_H_and_F_paths_from_folder(self):
        files_H = list(self.filepaths["folder_H_images"].glob("*.dcm"))
        files_F = list(self.filepaths["folder_F_images"].glob("*.dcm"))
        
        files_H_with_extras = [extract_extras_from_abspath(file) for file in files_H[:]]
        files_F_with_extras = [extract_extras_from_abspath(file) for file in files_F[:]]
        df_images_H = pd.DataFrame(files_H_with_extras)
        df_images_F = pd.DataFrame(files_F_with_extras)
        return df_images_H, df_images_F

    def load_autoscoRA_lables(self):
        df_autoscoRA_labels_H = pd.read_csv(self.filepaths["df_autoscoRA_labels_H"])
        df_autoscoRA_labels_F = pd.read_csv(self.filepaths["df_autoscoRA_labels_F"])
        return df_autoscoRA_labels_H, df_autoscoRA_labels_F 


    def load_everything(self):
        self.df_lm_labels_H, self.df_lm_labels_F = self.load_H_and_F_landmark_df()
        self.df_images_H, self.df_images_F = self.load_H_and_F_paths_from_folder()
        self.df_autoscoRA_labels_H, self.df_autoscoRA_labels_F = self.load_autoscoRA_lables()
        
        self.df_images_and_landmarks_F = pd.merge(self.df_images_F, self.df_lm_labels_F, on="filename", how="inner")
        self.df_images_and_landmarks_H = pd.merge(self.df_images_H, self.df_lm_labels_H, on="filename", how="inner")

        p = self.filepaths["training_test_splits_json_H"]
        if p != None:
            with open(p, 'r') as fp:
                data = json.load(fp)
            self.splits_dict_H = data
        else: 
            self.splits_dict_H = None
            
            
        p = self.filepaths["training_test_splits_json_F"]
        if p != None:
            with open(p, 'r') as fp:
                data = json.load(fp)
            self.splits_dict_F = data
        else: 
            self.splits_dict_F = None
            
        return None
    
    # --------------------------------------------------------------
    # Helper method: Extract image paths and landmark arrays from df
    # --------------------------------------------------------------
    def _extract_image_paths_and_landmarks(self, df_subset: pd.DataFrame):
        """
        Given a subset of self.df_images_and_landmarks_H (or F),
        return a list of image-path strings and a 2D numpy array
        of landmark coordinates of shape (num_images, num_landmarks, 2).
        """
        # Convert the 'image' column (a Path object) to string paths
        image_paths = df_subset["image"].apply(str).to_list()

        # Build the landmarks array
        # ra_utils.data.data_utils.extract_landmarks_from_df() presumably
        # expects a DataFrame with columns for each landmark_X and Y, etc.
        # We'll do exactly what your snippet suggests:
        landmarks_list = [
            ra_utils.data.data_utils.extract_landmarks_from_df(
                df_subset, image_idx=i
            ) 
            for i, _row in df_subset.iterrows()
        ]
        # Convert to NumPy, cast to uint16 (if that’s your preference),
        # and flip last axis if needed
        landmarks_array = np.array(landmarks_list, dtype=np.uint16)
        landmarks_array = np.flip(landmarks_array, axis=-1)  # e.g. (x,y) -> (y,x)

        return image_paths, landmarks_array

    # --------------------------------------------------------------
    #  get_landmarks_dataset_H
    # --------------------------------------------------------------
    def get_landmarks_dataset_H(self):
        """
        Returns the dataset splits for the 'hands' data as:
            image_paths_train,
            image_paths_test1,
            image_paths_test2,
            landmarks_train,
            landmarks_test1,
            landmarks_test2
        """
        if self.splits_dict_H is None:
            raise ValueError("No splits_dict_H loaded! Provide training_test_splits_json_H.")
        
        # Expecting that your JSON has keys: "training", "test1", "test2"
        if not all(k in self.splits_dict_H for k in ["training","test1","test2"]):
            raise ValueError("splits_dict_H is missing the required keys: 'training', 'test1', 'test2'")

        # 1) Filter DataFrame for each subset
        train_filenames = self.splits_dict_H["training"]
        test1_filenames = self.splits_dict_H["test1"]
        test2_filenames = self.splits_dict_H["test2"]

        df_train = self.df_images_and_landmarks_H[
            self.df_images_and_landmarks_H["filename"].isin(train_filenames)
        ]
        df_test1 = self.df_images_and_landmarks_H[
            self.df_images_and_landmarks_H["filename"].isin(test1_filenames)
        ]
        df_test2 = self.df_images_and_landmarks_H[
            self.df_images_and_landmarks_H["filename"].isin(test2_filenames)
        ]

        # 2) Extract image paths & landmarks for each subset
        image_paths_train, landmarks_train = self._extract_image_paths_and_landmarks(df_train)
        image_paths_test1, landmarks_test1 = self._extract_image_paths_and_landmarks(df_test1)
        image_paths_test2, landmarks_test2 = self._extract_image_paths_and_landmarks(df_test2)

        # 3) Return them
        return (
            image_paths_train,
            image_paths_test1,
            image_paths_test2,
            landmarks_train,
            landmarks_test1,
            landmarks_test2
        )

    # --------------------------------------------------------------
    #  get_landmarks_dataset_H_CV
    # --------------------------------------------------------------
    def get_landmarks_dataset_H_CV(self):
        """
        Returns cross-validation folds for 'hands' data.
        Each fold is a tuple:
            (train_image_paths, train_landmarks, test_image_paths, test_landmarks)
        
        So the final return is a list of those tuples, one entry per fold.
        """
        if self.splits_dict_H is None:
            raise ValueError("No splits_dict_H loaded! Provide training_test_splits_json_H.")
        
        if "cv" not in self.splits_dict_H:
            raise ValueError("splits_dict_H has no 'cv' key for cross-validation folds.")
        
        folds_data = []
        for fold_idx, fold in enumerate(self.splits_dict_H["cv"]):
            if not all(k in fold for k in ["train","test"]):
                raise ValueError(f"Fold {fold_idx} is missing 'train' or 'test' keys.")

            # Filenames for train/test
            train_filenames = fold["train"]
            test_filenames  = fold["test"]

            # Filter the main DataFrame
            df_train = self.df_images_and_landmarks_H[
                self.df_images_and_landmarks_H["filename"].isin(train_filenames)
            ]
            df_test = self.df_images_and_landmarks_H[
                self.df_images_and_landmarks_H["filename"].isin(test_filenames)
            ]

            # Extract image paths & landmarks
            train_image_paths, train_landmarks = self._extract_image_paths_and_landmarks(df_train)
            test_image_paths, test_landmarks = self._extract_image_paths_and_landmarks(df_test)

            # Save the fold’s data
            folds_data.append((train_image_paths, train_landmarks,
                               test_image_paths, test_landmarks))

        return folds_data    
    
    # --------------------------------------------------------------
    #  get_landmarks_dataset_F
    # --------------------------------------------------------------
    def get_landmarks_dataset_F(self):
        """
        Returns the dataset splits for the 'feet' data as:
            image_paths_train,
            image_paths_test1,
            image_paths_test2,
            landmarks_train,
            landmarks_test1,
            landmarks_test2
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

        image_paths_train, landmarks_train   = self._extract_image_paths_and_landmarks(df_train)
        image_paths_test1, landmarks_test1  = self._extract_image_paths_and_landmarks(df_test1)
        image_paths_test2, landmarks_test2  = self._extract_image_paths_and_landmarks(df_test2)

        return (
            image_paths_train,
            image_paths_test1,
            image_paths_test2,
            landmarks_train,
            landmarks_test1,
            landmarks_test2
        )

    # --------------------------------------------------------------
    #  get_landmarks_dataset_F_CV
    # --------------------------------------------------------------
    def get_landmarks_dataset_F_CV(self):
        """
        Returns cross-validation folds for 'feet' data.
        Each fold is a tuple:
            (train_image_paths, train_landmarks, test_image_paths, test_landmarks)
        
        So the final return is a list of those tuples, one entry per fold.
        """
        if self.splits_dict_F is None:
            raise ValueError("No splits_dict_F loaded! Provide training_test_splits_json_F.")
        
        if "cv" not in self.splits_dict_F:
            raise ValueError("splits_dict_F has no 'cv' key for cross-validation folds.")

        folds_data = []
        for fold_idx, fold in enumerate(self.splits_dict_F["cv"]):
            if not all(k in fold for k in ["train","test"]):
                raise ValueError(f"Fold {fold_idx} is missing 'train' or 'test' keys.")

            train_filenames = fold["train"]
            test_filenames  = fold["test"]

            df_train = self.df_images_and_landmarks_F[
                self.df_images_and_landmarks_F["filename"].isin(train_filenames)
            ]
            df_test = self.df_images_and_landmarks_F[
                self.df_images_and_landmarks_F["filename"].isin(test_filenames)
            ]

            train_image_paths, train_landmarks = self._extract_image_paths_and_landmarks(df_train)
            test_image_paths, test_landmarks   = self._extract_image_paths_and_landmarks(df_test)

            folds_data.append((train_image_paths, train_landmarks,
                               test_image_paths, test_landmarks))

        return folds_data