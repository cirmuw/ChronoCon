import yaml
from importlib import resources
import pandas as pd
import json
from typing import Literal, List

# Add ID for merging:
def double_scoring_make_merge_id(row):
    a = row["id_nr"]
    b = row["RO_datum"]
    c = row["bodypart_manual"]
    d = row["laterality_manual"]
    return f"{a}_{b}_{c}_{d}"

def limit_treatment_number(x, 
                            limit=5, 
                            limit_treatment: Literal["over_limit_to_NA", 
                                                     "over_limit_to_limit",
                                                     "None",
                                                     "over_limit_to_limit_plus_1"
                                                     ] = "over_limit_to_limit"):
    if pd.isna(x):
        return x
    else: 
        try: 
            x = int(x)
        except Exception as e:
            print(f"{x=} could not be converted to int -> pd.NA") 
            return pd.NA
        if x > limit: 
            if limit_treatment ==  "over_limit_to_limit": 
                return limit
            elif limit_treatment ==  "over_limit_to_NA":
                return pd.NA
            elif limit_treatment == "None":
                return x
            elif limit_treatment == "over_limit_to_limit_plus_1":
                return limit + 1
            else: 
                raise NotImplementedError(f"{x=} {limit_treatment=}")
        else: 
            return x

def _sum_scores_Shap(row: dict, 
                     scores: List[str], 
                     limit=5, 
                     limit_treatment: Literal["over_limit_to_NA", "over_limit_to_limit", "None", "over_limit_to_limit_plus_1"] = "over_limit_to_limit"):
    result = 0
    for score in scores:
        r = row[score]
        try:
            r = limit_treatment_number(r, limit=limit, limit_treatment=limit_treatment)
        except Exception as e: 
            print(f"{score=}  {r=}")
            raise e        
        result += r
    return result


def add_bodypart_totals(
    df: pd.DataFrame,
    col: str,                     # e.g. "cw_ERO_HF_LR"
    metric: str,                     # "ERO" or "JSN"
    prefix = "my_"
) -> pd.DataFrame:
    """
    Adds/over-writes two columns per metric:
        my_<metric>_H   ─ summed L+R hands   per patient-visit
        my_<metric>_F   ─ summed L+R feet    per patient-visit
    Running the function repeatedly is safe.
    """
    df = df.copy()

    new_cols = [f"{prefix}{metric}_H", f"{prefix}{metric}_F"]

    # ---- drop any previous versions to dodge the merge-overlap error ----
    df.drop(columns=[c for c in new_cols if c in df.columns], inplace=True)

    # ---- compute fresh per-visit sums -----------------------------------
    pivot = (
        df.groupby(["patientId_visitDate", "bodypart_manual"])[col]
          .sum()
          .unstack("bodypart_manual")                # columns: H / F
          .rename(columns={"H": new_cols[0], "F": new_cols[1]})
    )

    # ---- merge back (no overlap possible now) ---------------------------
    df = df.merge(pivot, on="patientId_visitDate", how="left")

    return df

def add_JSN_ERO_sums(df, 
                     limit_treatment: Literal['over_limit_to_NA', 'over_limit_to_limit', "over_limit_to_limit_plus_1",'None'] = "over_limit_to_limit",
                     column_prefix = "cw_"
                     ):
    df = df.copy()
    
    with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv").open("r") as f:
        df_score_roi_matching = pd.read_csv(f)

    # Make sure the working columns exist with the right dtype *once*    #
    for col in (f"{column_prefix}ERO_HF_LR", f"{column_prefix}JSN_HF_LR"):
        if col not in df.columns:
            df[col] = pd.Series(dtype="Int64")   # nullable integer
        else:
            df[col] = df[col].astype("Int64", errors="ignore")

    # H ERO
    m = (df_score_roi_matching["region"] == "H") & (df_score_roi_matching["ERO_or_JSN"] == "ERO")
    relevant_scores_H = list(df_score_roi_matching[m]["score_name"])
    mask_H = df["bodypart_manual"] == "H"
    df.loc[mask_H, f"{column_prefix}ERO_HF_LR"] = df.loc[mask_H].apply(
        lambda row: _sum_scores_Shap(row, relevant_scores_H, limit=5, limit_treatment=limit_treatment),
        axis=1
    ).astype("Int64") 

    # F ERO
    m = (df_score_roi_matching["region"] == "F") & (df_score_roi_matching["ERO_or_JSN"] == "ERO")
    relevant_scores_F = list(df_score_roi_matching[m]["score_name"])
    mask_F = df["bodypart_manual"] == "F"
    df.loc[mask_F, f"{column_prefix}ERO_HF_LR"] = df.loc[mask_F].apply(
        lambda row: _sum_scores_Shap(row, relevant_scores_F, limit=5, limit_treatment=limit_treatment),
        axis=1
    ).astype("Int64") 

    # H JSN
    m = (df_score_roi_matching["region"] == "H") & (df_score_roi_matching["ERO_or_JSN"] == "JSN")
    relevant_scores_H = list(df_score_roi_matching[m]["score_name"])
    mask_H = df["bodypart_manual"] == "H"
    df.loc[mask_H, f"{column_prefix}JSN_HF_LR"] = df.loc[mask_H].apply(
        lambda row: _sum_scores_Shap(row, relevant_scores_H, limit=4, limit_treatment=limit_treatment),
        axis=1
    ).astype("Int64") 

    # F JSN
    m = (df_score_roi_matching["region"] == "F") & (df_score_roi_matching["ERO_or_JSN"] == "JSN")
    relevant_scores_F = list(df_score_roi_matching[m]["score_name"])
    mask_F = df["bodypart_manual"] == "F"
    df.loc[mask_F, f"{column_prefix}JSN_HF_LR"] = df.loc[mask_F].apply(
        lambda row: _sum_scores_Shap(row, relevant_scores_F, limit=4, limit_treatment=limit_treatment),
        axis=1
    ).astype("Int64") 


    # --- 2) apply for ERO and JSN --------------------------------------------
    df = add_bodypart_totals(df, f"{column_prefix}ERO_HF_LR", "ERO", prefix=column_prefix)
    df = add_bodypart_totals(df, f"{column_prefix}JSN_HF_LR", "JSN", prefix=column_prefix)

    # # --- 3) grand totals -----------------------------------------------------
    # Total ERO
    df[f"{column_prefix}ERO"] = df[f"{column_prefix}ERO_F"] + df[f"{column_prefix}ERO_H"] 

    # Total JSN
    df[f"{column_prefix}JSN"] = df[f"{column_prefix}JSN_F"] + df[f"{column_prefix}JSN_H"] 

    # Total SvdH
    df[f"{column_prefix}SvdH"] = df[f"{column_prefix}JSN"] + df[f"{column_prefix}ERO"]

    return df
