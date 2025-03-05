
import numpy as np
from sklearn.model_selection import GroupKFold


# This splits the data and keeps the patients in same set
#
# Usage example 
#
#  Make splits file 
# df_splits_input = dataHandler.df_lm_labels_F[["filename"]].copy()
# df_splits_input["patient_id"] = df_splits_input["filename"].apply(lambda x: x.split("_")[0])
#
# split_dict = generate_split_dictionary(df_splits_input, 
#                                        cw_splits=3,
#                                        train_val_test_split_proportions=(0.6,0.2,0.2),
#                                        random_seed=42)

# import json
# dst = "/home/clemens/data/AutoPIX_cirdata_local/projects__autoscora/landmark_data/splits/splits_F_25-03-05.json"
# with open(dst, 'w') as fp:
#     json.dump(split_dict, fp, sort_keys=True, indent=4)
#
# I used this to generate the splits for the landmarks, but I then added 
# for the CV the original splits used by Thomas and Paul 
# 
# TODO add maybe stratification later when we have data

def generate_split_dictionary(
    df_splits_input,
    cw_splits=5, 
    train_val_test_split_proportions=(0.6, 0.2, 0.2),
    random_seed=42
):
    """
    Returns a dictionary with two major parts:
      1) 'cv': a list of cross-validation folds, where each element is a dict
         containing 'train' and 'test' keys with lists of filenames.
         - Each fold is based on patients (i.e., GroupKFold).
      2) The final train/test1/test2 split is also performed on the patient level.

    Parameters
    ----------
    df_splits_input : pd.DataFrame
        A DataFrame containing (at least) two columns:
          - "filename" whose values serve as unique IDs for images
          - "patient_id" which groups images by patient
    cw_splits : int, default=5
        Number of cross-validation folds to generate.
    train_val_test_split_proportions : tuple of floats, default=(0.6, 0.2, 0.2)
        Proportions used for final dataset splitting.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary with the structure:
            {
                "cv": [
                    {"train": [...], "test": [...]},
                    {"train": [...], "test": [...]},
                    ...
                ],
                "training": [...],
                "test1": [...],
                "test2": [...]
            }

        where "cv" is a list of cross-validation folds
        and "training"/"test1"/"test2" are the final splits.
    """

    # Ensure the proportions sum to 1.0
    assert abs(sum(train_val_test_split_proportions) - 1.0) < 1e-9, \
        "train_val_test_split_proportions must sum to 1.0"

    # -------------------------------------------------
    # 1) Unique patients (group identifiers)
    # -------------------------------------------------
    # We'll split on patient_id, not on every filename
    unique_patients = df_splits_input["patient_id"].unique()
    n_patients = len(unique_patients)

    # Shuffle the patient IDs
    rng = np.random.RandomState(random_seed)
    shuffled_patients = rng.permutation(unique_patients)

    # -------------------------------------------------
    # 2) Cross-validation with GroupKFold
    # -------------------------------------------------
    # Prepare a GroupKFold to ensure that all samples from each patient end up
    # in the same fold
    gkf = GroupKFold(n_splits=cw_splits)

    cv_splits = []
    # We need to pass 'X' (features) and 'y' (labels) to GroupKFold, but we only 
    # use them for indexing. The important part is the 'groups' argument.
    # For convenience, let's create an array from the shuffled patient list 
    # so we can do indexing easily.
    # 
    # Trick: We'll just index from 0...n_patients for the patients in "shuffled_patients"
    # and then map each index back to a patient ID and then to the actual filenames.
    # 
    # We need a dictionary from patient_id -> index in the 'shuffled_patients' array.
    patient_to_idx = {p: i for i, p in enumerate(shuffled_patients)}
    
    # For each row in df_splits_input, store the integer index of its patient
    # (the one in 'shuffled_patients').
    df_splits_input["group_idx"] = df_splits_input["patient_id"].map(patient_to_idx)
    
    # We'll "simulate" data X = range(n_patients), y = 0 or something,
    # but the groups are indeed the group_idx.
    # Actually, we need an array where each sample is a single row from df,
    # but GroupKFold is typically at the row level. If we do so, we might end
    # up splitting the same patient across folds if we apply it directly.
    # 
    # Instead, let's do it at the PATIENT level: We'll pass X = range(n_patients)
    # and groups = range(n_patients) so that each patient is a single "sample".
    # Then for each fold, we gather the patient IDs (train, test).
    # 
    # This approach means each "split" gives us sets of *patients* in train/test.
    # Then we map those to actual filenames. 
    
    X = np.arange(n_patients)
    groups = X  # each patient is its own group
    
    for train_patient_idxs, test_patient_idxs in gkf.split(X, y=None, groups=groups):
        # The actual patient IDs
        train_patients = shuffled_patients[train_patient_idxs]
        test_patients  = shuffled_patients[test_patient_idxs]

        # Filenames that belong to those train/test patients
        train_filenames = df_splits_input[
            df_splits_input["patient_id"].isin(train_patients)
        ]["filename"].tolist()

        test_filenames = df_splits_input[
            df_splits_input["patient_id"].isin(test_patients)
        ]["filename"].tolist()

        cv_splits.append({
            "train": train_filenames,
            "test":  test_filenames
        })
    
    # -------------------------------------------------
    # 3) Final train/test1/test2 split at patient level
    # -------------------------------------------------
    train_prop, test1_prop, test2_prop = train_val_test_split_proportions

    train_end = int(train_prop * n_patients)
    test1_end = int(test1_prop * n_patients) + train_end

    train_patients_final = shuffled_patients[:train_end]
    test1_patients_final = shuffled_patients[train_end:test1_end]
    test2_patients_final = shuffled_patients[test1_end:]

    # Gather corresponding filenames for each set
    df_train = df_splits_input[df_splits_input["patient_id"].isin(train_patients_final)]
    df_test1 = df_splits_input[df_splits_input["patient_id"].isin(test1_patients_final)]
    df_test2 = df_splits_input[df_splits_input["patient_id"].isin(test2_patients_final)]
    
    # Convert to lists of filenames
    training_filenames = df_train["filename"].tolist()
    test1_filenames = df_test1["filename"].tolist()
    test2_filenames = df_test2["filename"].tolist()

    # -------------------------------------------------
    # 4) Build and return the final dictionary
    # -------------------------------------------------
    result = {
        "cv": cv_splits,
        "training": training_filenames,
        "test1": test1_filenames,
        "test2": test2_filenames
    }
    return result