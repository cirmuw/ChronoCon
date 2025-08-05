from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import re


# Optional pydicom import for PixelSpacing (only used if get_pixel_spacing=True)
try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except Exception:  # pragma: no cover
    pydicom = None
    InvalidDicomError = Exception

import pandas as pd
import numpy as np
import json
import yaml
import os


import ra_utils
import ra_utils.data.data_utils
from ra_utils.data.data_utils import (
    extract_extras_from_filename, 
    extract_extras_from_abspath
)



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




class DataHandler_CR_autoscoRA_generic(object):
    """
    Generic data handler that covers the shared logic for H/F (or any other) datasets.

    It loads:
      • image file paths from a folder of DICOMs
      • a landmark CSV (with either header="infer" or header=None)
      • an optional JSON/YAML describing dataset splits (training/test1/test2 and/or cv)

    It can then produce train/test splits (with optional PixelSpacing) or CV folds.

    Notes
    -----
    * The parameter name `df_autoscoRA_labels_header` mirrors your old signature, but it
      is used here as the *landmark CSV header mode*. Consider renaming it to
      `df_lm_header` for clarity.
    * We assume your helper `extract_extras_from_abspath` returns a dict that includes
      at least keys `file_name` (the basename without extension) and `image` (absolute
      path to the file). If it does not provide `image`, we fall back to constructing it
      as `folder_images / filename`.
    * Landmark arrays are returned with last axis flipped (x, y) -> (y, x) to match the
      prior behavior.
    """

    def __init__(
        self,
        folder_images: str | Path,
        df_lm_labels: str | Path,
        training_test_splits_json: Optional[str | Path] = None,
        df_autoscoRA_labels_header: str | None = "infer",
    ) -> None:
        self.filepaths = dict(
            folder_images=Path(folder_images),
            df_lm_labels=Path(df_lm_labels),
            training_test_splits_json=training_test_splits_json,
        )
        self.df_lm_header = df_autoscoRA_labels_header  # kept for backward signature compatibility

        self._load_everything()

    # -------------------------
    # CSV / folder load helpers
    # -------------------------
    @staticmethod
    def _load_landmark_df(landmark_path: Path | str, header: str | None = "infer") -> pd.DataFrame:
        df = pd.read_csv(landmark_path, header=header)
        df = df.rename(columns={"img": "filename"})

        if header is None:
            # auto-generate column names: filename, landmark_0_0, landmark_0_1, ...
            column_names = ["filename"] + [
                f"landmark_{(i-1) // 2}_{(i-1) % 2}" for i in range(1, df.shape[1])
            ]
            df.columns = column_names

        # Drop rows where *all* landmark columns are zero
        if df.shape[1] > 1:
            m = ~(df.iloc[:, 1:] == 0).all(axis=1)
            df = df[m]
        return df

    def _load_paths_from_folder(self) -> pd.DataFrame:
        folder = self.filepaths["folder_images"]
        files = list(folder.glob("*.dcm"))
        files = [f for f in files if not f.name.startswith("._")]  # ignore macOS artifacts

        # Extract extras per file (relies on your existing helper)
        files_with_extras = [
            extract_extras_from_abspath(file, ending=".dcm", replace_ending=True) for file in files
        ]
        df_images = pd.DataFrame(files_with_extras).rename(columns={"file_name": "filename"})

        # Ensure we have an `image` column to point to the absolute path
        if "image" not in df_images.columns:
            df_images["image"] = [str(folder / f"{fn}.dcm") for fn in df_images["filename"].astype(str)]
        return df_images

    def _load_splits(self) -> Optional[dict]:
        p = self.filepaths["training_test_splits_json"]
        if p is None:
            return None
        data = read_dict_from_file(p)
        return data.copy() if data is not None else None

    def _load_everything(self) -> None:
        # Core dataframes
        self.df_lm_labels = self._load_landmark_df(self.filepaths["df_lm_labels"], header=self.df_lm_header)
        self.df_images = self._load_paths_from_folder()

        # Landmark base names (without -X/-Y/_0/_1 suffixes). We take every second column (= X),
        # assuming pairs (X, Y) or (_0, _1) are contiguous.
        self.landmark_names = (
            [re.sub(r"((-X)|(-Y)|(_0)|_1)$", "", c) for c in self.df_lm_labels.columns[1:]][::2]
            if self.df_lm_header == "infer"
            else None
        )

        # Merge images and landmarks on filename
        self.df_images_and_landmarks = pd.merge(
            self.df_images, self.df_lm_labels, on="filename", how="inner"
        )

        # Splits (may be None)
        self.splits_dict = self._load_splits()

    # --------------------------------------------------------------
    # Helper: extract image paths and landmark arrays from a subset
    # --------------------------------------------------------------
    def _extract_image_paths_and_landmarks(
        self,
        df_subset: pd.DataFrame,
        get_pixel_spacing: bool = False,
        landmark_names: Optional[List[str]] = None,
        pixel_spacing_default: Iterable[float] = (0.1, 0.1),
    ) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
        """
        Returns (image_paths, landmarks_array, pixel_spacing_array|None)
        * image_paths: list[str]
        * landmarks_array: np.ndarray of shape (N, L, 2), dtype uint16, with last axis flipped to (y, x)
        * pixel_spacing_array: np.ndarray of shape (N, 2), dtype float, or None if not requested
        """
        # Convert Path objects to string paths
        image_paths = df_subset["image"].apply(str).tolist()

        # Build landmarks array using your existing helper.
        # IMPORTANT: `extract_landmarks_from_df` uses `.loc[image_idx, ...]`, so we must pass
        # *label* indices, not positional indices. Using df_subset.index preserves labels
        # even if the DataFrame index is non-consecutive after merges/filters.
        landmarks_list = [
            ra_utils.data.data_utils.extract_landmarks_from_df(
                df_subset, image_idx=i, landmark_names=landmark_names
            )
            for i in df_subset.index
        ]
        landmarks_array = np.array(landmarks_list, dtype=np.uint16)
        # (x, y) -> (y, x) for downstream consumers that expect that order
        landmarks_array = np.flip(landmarks_array, axis=-1)

        pixel_spacing_array = None
        if get_pixel_spacing:
            pixel_spacings: List[List[float]] = []
            for dcm_path in df_subset["image"].tolist():
                ps = pixel_spacing_default
                if pydicom is not None:
                    try:
                        ds = pydicom.dcmread(str(dcm_path), force=True)
                        try:
                            ps = ds.PixelSpacing  # [RowSpacing, ColSpacing]
                        except Exception:
                            # keep default
                            pass
                    except Exception:
                        # invalid DICOM, keep default
                        pass
                pixel_spacings.append(ps)
            pixel_spacing_array = np.asarray(pixel_spacings, dtype=float)

        return image_paths, landmarks_array, pixel_spacing_array

    # --------------------------------------------------------------
    # Public API: explicit splits (training/test1/test2)
    # --------------------------------------------------------------
    def get_landmarks_dataset(
        self,
        get_pixel_spacing: bool = False,
        pixel_spacing_default: Iterable[float] = (0.1, 0.1),
    ) -> Tuple[
        List[str], List[str], List[str],
        np.ndarray, np.ndarray, np.ndarray,
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    ]:
        """
        Returns a 9-tuple analogous to the old H/F-specific methods:
            (
                image_paths_train,   image_paths_test1,   image_paths_test2,
                landmarks_train,     landmarks_test1,     landmarks_test2,
                pixel_spacing_train, pixel_spacing_test1, pixel_spacing_test2
            )
        Requires that self.splits_dict contains keys: "training", "test1", "test2".
        """
        if self.splits_dict is None:
            raise ValueError("No splits_dict loaded! Provide training_test_splits_json.")
        required = {"training", "test1", "test2"}
        if not required.issubset(self.splits_dict.keys()):
            missing = required - set(self.splits_dict.keys())
            raise ValueError(f"splits_dict is missing required keys: {sorted(missing)}")

        train_filenames = self.splits_dict.get("training") or []
        test1_filenames = self.splits_dict.get("test1") or []
        test2_filenames = self.splits_dict.get("test2") or []

        df_train = self.df_images_and_landmarks[self.df_images_and_landmarks["filename"].isin(train_filenames)]
        df_test1 = self.df_images_and_landmarks[self.df_images_and_landmarks["filename"].isin(test1_filenames)]
        df_test2 = self.df_images_and_landmarks[self.df_images_and_landmarks["filename"].isin(test2_filenames)]

        (image_paths_train, landmarks_train, pixel_spacing_train) = self._extract_image_paths_and_landmarks(
            df_train,
            get_pixel_spacing=get_pixel_spacing,
            landmark_names=self.landmark_names,
            pixel_spacing_default=pixel_spacing_default,
        )
        (image_paths_test1, landmarks_test1, pixel_spacing_test1) = self._extract_image_paths_and_landmarks(
            df_test1,
            get_pixel_spacing=get_pixel_spacing,
            landmark_names=self.landmark_names,
            pixel_spacing_default=pixel_spacing_default,
        )
        (image_paths_test2, landmarks_test2, pixel_spacing_test2) = self._extract_image_paths_and_landmarks(
            df_test2,
            get_pixel_spacing=get_pixel_spacing,
            landmark_names=self.landmark_names,
            pixel_spacing_default=pixel_spacing_default,
        )

        return (
            image_paths_train,
            image_paths_test1,
            image_paths_test2,
            landmarks_train,
            landmarks_test1,
            landmarks_test2,
            pixel_spacing_train,
            pixel_spacing_test1,
            pixel_spacing_test2,
        )

    # --------------------------------------------------------------
    # Public API: cross-validation folds
    # --------------------------------------------------------------
    def get_landmarks_dataset_CV(self) -> List[Tuple[List[str], np.ndarray, List[str], np.ndarray]]:
        """
        Returns CV folds. Each fold is a tuple:
            (train_image_paths, train_landmarks, test_image_paths, test_landmarks)
        Expectation: self.splits_dict has a key "cv" mapping to a list of {train: [...], test: [...]}.
        """
        if self.splits_dict is None:
            raise ValueError("No splits_dict loaded! Provide training_test_splits_json.")
        if "cv" not in self.splits_dict:
            raise ValueError("splits_dict has no 'cv' key for cross-validation folds.")

        folds_data: List[Tuple[List[str], np.ndarray, List[str], np.ndarray]] = []
        for fold_idx, fold in enumerate(self.splits_dict["cv"]):
            if not {"train", "test"}.issubset(fold):
                raise ValueError(f"Fold {fold_idx} missing 'train' or 'test' keys.")

            train_filenames = fold["train"] or []
            test_filenames = fold["test"] or []

            df_train = self.df_images_and_landmarks[self.df_images_and_landmarks["filename"].isin(train_filenames)]
            df_test = self.df_images_and_landmarks[self.df_images_and_landmarks["filename"].isin(test_filenames)]

            (train_image_paths, train_landmarks, _) = self._extract_image_paths_and_landmarks(
                df_train, get_pixel_spacing=False, landmark_names=self.landmark_names
            )
            (test_image_paths, test_landmarks, _) = self._extract_image_paths_and_landmarks(
                df_test, get_pixel_spacing=False, landmark_names=self.landmark_names
            )

            folds_data.append((train_image_paths, train_landmarks, test_image_paths, test_landmarks))

        return folds_data


# # -----------------------------
# # Optional: Back-compat adapter
# # -----------------------------
# class DataHandler_CR_autoscoRA(object):
#     """
#     Thin compatibility wrapper that recreates the original H/F API using two
#     instances of the generic handler. Drop-in replacement if you want to avoid
#     refactoring call sites immediately.
#     """

#     def __init__(
#         self,
#         folder_H_images: str | Path,
#         folder_F_images: str | Path,
#         df_lm_labels_H: str | Path,
#         df_lm_labels_F: str | Path,
#         df_autoscoRA_labels_F: str | Path | None = None,  # kept but unused here
#         df_autoscoRA_labels_H: str | Path | None = None,  # kept but unused here
#         training_test_splits_json_H: Optional[str | Path] = None,
#         training_test_splits_json_F: Optional[str | Path] = None,
#         df_autoscoRA_labels_F_header: str | None = "infer",
#         df_autoscoRA_labels_H_header: str | None = "infer",
#     ) -> None:
#         self.H = DataHandler_CR_autoscoRA_generic(
#             folder_images=folder_H_images,
#             df_lm_labels=df_lm_labels_H,
#             training_test_splits_json=training_test_splits_json_H,
#             df_autoscoRA_labels_header=df_autoscoRA_labels_H_header,
#         )
#         self.F = DataHandler_CR_autoscoRA_generic(
#             folder_images=folder_F_images,
#             df_lm_labels=df_lm_labels_F,
#             training_test_splits_json=training_test_splits_json_F,
#             df_autoscoRA_labels_header=df_autoscoRA_labels_F_header,
#         )

#     # Mirror the old methods
#     def get_landmarks_dataset_H(self, get_pixel_spacing: bool = False, pixel_spacing_default=(0.1, 0.1)):
#         return self.H.get_landmarks_dataset(get_pixel_spacing=get_pixel_spacing, pixel_spacing_default=pixel_spacing_default)

#     def get_landmarks_dataset_F(self, get_pixel_spacing: bool = False, pixel_spacing_default=(0.1, 0.1)):
#         return self.F.get_landmarks_dataset(get_pixel_spacing=get_pixel_spacing, pixel_spacing_default=pixel_spacing_default)

#     def get_landmarks_dataset_H_CV(self):
#         return self.H.get_landmarks_dataset_CV()

#     def get_landmarks_dataset_F_CV(self):
#         return self.F.get_landmarks_dataset_CV()

#     # Expose a few fields for compatibility, if needed
#     @property
#     def df_images_and_landmarks_H(self):
#         return self.H.df_images_and_landmarks

#     @property
#     def df_images_and_landmarks_F(self):
#         return self.F.df_images_and_landmarks

#     @property
#     def landmark_names_H(self):
#         return self.H.landmark_names

#     @property
#     def landmark_names_F(self):
#         return self.F.landmark_names

