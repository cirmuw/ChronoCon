from pathlib import Path
import pandas as pd

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
                 df_autoscoRA_labels_H = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_data/autoscoRA_hands.csv"
                 ):
        self.filepaths = dict(
                 folder_H_images = Path(folder_H_images),
                 folder_F_images = Path(folder_F_images),
                 df_lm_labels_H = Path(df_lm_labels_H),
                 df_lm_labels_F = Path(df_lm_labels_F),
                 df_autoscoRA_labels_F = Path(df_autoscoRA_labels_F),
                 df_autoscoRA_labels_H = Path(df_autoscoRA_labels_H)
        )

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

        # TODO merge other labels and images

