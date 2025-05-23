import yaml
from importlib import resources
import pandas as pd
import json
from typing import Literal, List



def filter_valid_extrapolated_SHS_sums(df_summed: pd.DataFrame, scores: List[str]) -> pd.DataFrame:
    """
    Motivation: 
    Some scores are not valid (e.g. surgery, bad view, ...). If to many joints could not be scored, 
    then these should not be included in the evaluation. 
    
    A more elaborate inclusion scheme can be used later.
    """

    # This could be used for valid imputations...    
    # with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv") as f:
    #     df_scores_meta = pd.read_csv(f)
    # H_ERO_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "ERO") & (df_scores_meta["region"] == "H")]["score_name"].unique())
    # F_ERO_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "ERO") & (df_scores_meta["region"] == "F")]["score_name"].unique())
    # H_JSN_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "JSN") & (df_scores_meta["region"] == "H")]["score_name"].unique())
    # F_JSN_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "JSN") & (df_scores_meta["region"] == "F")]["score_name"].unique())
        
    m = (df_summed["n_valid_ERO"] + df_summed["n_valid_JSN"]) > (len(set(scores)) * 0.5)
    return df_summed[m]
    

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

#^^^^^^^^^^^^^^#
##### OLD ######


def sum_scores_df(df: pd.DataFrame,
                  scores_to_sum: List[str]) -> pd.DataFrame:
    """
    Aggregate labels & preds across the requested score_types and both sides (L / R),
    returning one line per patient-date together with a count of how many valid
    samples contributed to each sum for ERO and JSN.

    Expected columns in *df*
        - patientId_date          (str)   – remaining grouping key
        - score_type              (str)
        - JSN_or_ERO              (str)   – either 'JSN' or 'ERO'
        - labels, preds           (numeric, NaN allowed)

    The function:
        1. keeps only rows whose score_type is in *scores_to_sum*
        2. treats rows as “valid” when **both** labels and preds are *not* NaN
        3. sums labels & preds per patientId_date
        4. counts, in the valid subset, how many rows belong to each JSN_or_ERO class
        5. emits columns
              patientId_date, labels_summed, preds_summed,
              n_valid_ERO,   n_valid_JSN
    """
    # 1) filter the score types we care about
    df_filt = df[df["score_type"].isin(scores_to_sum)].copy()
    # get limits of numbers
    with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv") as f:
        df_scores_meta = pd.read_csv(f)
        limits_dct = df_scores_meta[["score_name", "limit"]].set_index("score_name").to_dict()["limit"]

    df_filt["limit"] = df_filt["score_type"].apply(lambda x: int(limits_dct[x]))
    # 2) mark rows that actually contain usable numbers



    df_filt["is_valid"] = ( (~df_filt["labels"].isna()) & 
                            (~df_filt["preds"].isna()) & 
                            (df_filt["labels"] <= df_filt["score_type"].apply(lambda x: limits_dct[x]))
                          )

    # 3) sums (NaNs are ignored because they were filtered out in is_valid)
    sums = (
        df_filt.groupby("patientId_date")[["labels", "preds", "limit"]]
               .sum()
               .rename(columns={"labels": "labels_summed",
                                "preds":  "preds_summed",
                                "limit":  "limit_summed"})
    )

    # 4) how many valid observations per JSN / ERO
    counts = (
        df_filt.loc[df_filt["is_valid"]]
               .groupby(["patientId_date", "JSN_or_ERO"])
               .size()
               .unstack(fill_value=0)          # one column per JSN/ERO
               .rename(columns={"ERO": "n_valid_ERO", "JSN": "n_valid_JSN"})
    )

    # ensure both columns exist even if one class is absent
    for col in ["n_valid_ERO", "n_valid_JSN"]:
        if col not in counts.columns:
            counts[col] = 0

    return sums.reset_index()

    # # 5) put everything together
    # out = (sums.join(counts)[
    #           ["labels_summed", "preds_summed", "limit_summed", "n_valid_ERO", "n_valid_JSN"]
    #        ]
    #        .reset_index()
    #        .astype({"n_valid_ERO": int, "n_valid_JSN": int})
    # )
    # return out


def sum_scores_df_JSN_H(df: pd.DataFrame):
    H_JSN = ["PIPII", "PIPIII", "PIPIV", "PIPV", 
             "MCPI", "MCPII", "MCPIII", "MCPIV", "MCPV",
             "CMCIII", "CMCIV", "CMCV",
             "Rad_Carp", "Sca_Cap", "Tra_Sca"]
    return sum_scores_df(df, scores_to_sum=H_JSN)


def sum_scores_df_JSN_F(df: pd.DataFrame):
    F_JSN = ["MTPI", "MTPII", "MTPIII", "MTPIV", "MTPV", "IP"]
    return sum_scores_df(df, scores_to_sum=F_JSN)


def sum_scores_df_ERO_F(df: pd.DataFrame):
    F_ERO = ["MTPIEP", "MTPIED", 
             "MTPIIEP", "MTPIIIEP", "MTPIVEP", "MTPVEP",
             "MTPIIED", "MTPIIIED", "MTPIVED", "MTPVED",
             "IPEP", "IPED"
             ]
    return sum_scores_df(df, scores_to_sum=F_ERO)



def sum_scores_df_ERO_H(
    df: pd.DataFrame,
    limit_treatment_ED: Literal["cap E_D sum to 5", "E_D_mean"] = "cap E_D sum to 5"
) -> pd.DataFrame:
    """
    Aggregate hand ERO scores.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (see doc-string of ``sum_scores_df`` for the expected
        schema).
    limit_treatment_ED : {"cap E_D sum to 5", "E_D_mean"}
        • "cap E_D sum to 5"  – the sum of proximal (EP) + distal (ED) parts of
          each joint is clipped at 5.  
        • "E_D_mean"         – the arithmetic mean of EP and ED is taken
          (i.e. the summed value is divided by 2).

    Returns
    -------
    pd.DataFrame
        One row per *patientId_date* with the columns
        ``labels_summed``, ``preds_summed``, ``limit_summed``,
        ``n_valid_ERO`` and ``n_valid_JSN``.
    """

    # ---------------------------------------------------------------------
    # 1.  JOINT PAIRS (EP + ED)
    # ---------------------------------------------------------------------
    joints = [
        "IPI", "PIPII", "PIPIII", "PIPIV", "PIPV",
        "MCPI", "MCPII", "MCPIII", "MCPIV", "MCPV"
    ]
    score_pairs: List[List[str]] = [[f"{j}ED", f"{j}EP"] for j in joints]

    dfs_pairs = []
    for pair in score_pairs:
        # Sum ED + EP for one joint
        df_pair = sum_scores_df(df, scores_to_sum=pair)

        if limit_treatment_ED == "cap E_D sum to 5":
            # Clip the pair-wise totals at 5
            for col in ["labels_summed", "preds_summed", "limit_summed"]:
                df_pair[col] = df_pair[col].clip(upper=5*2)

        elif limit_treatment_ED == "E_D_mean":
            # Convert the sum to a mean of the two components
            for col in ["labels_summed", "preds_summed", "limit_summed"]:
                df_pair[col] = df_pair[col] / 2

        else:           # should never happen; guard for forwards-compat
            raise NotImplementedError(f"Unknown mode: {limit_treatment_ED}")

        dfs_pairs.append(df_pair)

    df_summed_pairs = pd.concat(dfs_pairs, ignore_index=True)

    # ---------------------------------------------------------------------
    # 2.  SINGLE-SITE SCORES (already limited to ≤ 5 individually)
    # ---------------------------------------------------------------------
    H_ERO_singles = [
        "Base_MCIE", "LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"
    ]
    df_summed_singles = sum_scores_df(df, scores_to_sum=H_ERO_singles)

    # ---------------------------------------------------------------------
    # 3.  FINAL AGGREGATION  (pairs  +  singles)
    # ---------------------------------------------------------------------
    df_all = pd.concat([df_summed_pairs, df_summed_singles], ignore_index=True)

    df_summed = (
        df_all
        .groupby("patientId_date", as_index=False)
        .agg({
            "labels_summed": "sum",
            "preds_summed":  "sum",
            "limit_summed":  "sum",
            #"n_valid_ERO":   "sum",
            #"n_valid_JSN":   "sum",
        })
        #.astype({"n_valid_ERO": int, "n_valid_JSN": int})
    )

    return df_summed








# def max_possible_score(scores_to_sum = List[str]):
#     with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv") as f:
#         df_scores_meta = pd.read_csv(f)
#         limits_dct = df_scores_meta[["score_name", "limit"]].set_index("score_name").to_dict()["limit"]
#     max_score_possible_out = sum([limits_dct[s] for s in scores_to_sum])
#     return max_score_possible_out * 2 # L + R

# def sum_and_extrapolate_scores_df(df: pd.DataFrame,
#                                   scores_to_sum: List[str]) -> pd.DataFrame:
#     df_summed = sum_scores_df(df, scores_to_sum = scores_to_sum)
#     m = max_possible_score(scores_to_sum)
#     df_summed["preds_summed_extrapolated"] = df_summed["preds_summed"] * m / df_summed["limit_summed"]
#     df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * m / df_summed["limit_summed"]
#     return df_summed


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def max_possible_score(
    scores_to_sum: List[str],
    collapse_ED_EP_pairs: bool = False,      # <-- NEW
) -> int:
    """
    Compute the theoretical maximum that *one* observation could reach.

    Parameters
    ----------
    scores_to_sum : list[str]
        All score names that will later be aggregated.
    collapse_ED_EP_pairs : bool, default False
        When ``True`` an 'EP' + 'ED' combination of the same *hand* joint is
        counted only once with a limit of 5 pts (instead of twice with 10 pts).
        Needed for the new hand-erosion logic; leave «False» for every other
        score set (feet, JSN, …).

    Returns
    -------
    int
        Maximum possible score for *both* sides (left + right).
    """
    with resources.files("ra_utils.resources.scores_metadata") \
            .joinpath("roi_scores_matching.csv") as f:
        df_scores_meta = pd.read_csv(f)

    limits = (
        df_scores_meta[["score_name", "limit"]]
        .set_index("score_name")["limit"]
        .to_dict()
    )

    if not collapse_ED_EP_pairs:
        # original behaviour
        per_side = sum(limits[s] for s in scores_to_sum)
        return per_side * 2                      # L + R

    # ---- new branch: collapse ED / EP PAIRS (hand only) ------------------ #
    already_counted = set()
    per_side = 0

    for s in scores_to_sum:
        if s.endswith(("ED", "EP")):             # potential half-pair
            base = s[:-2]                        # strip ED/EP suffix
            if base not in already_counted:
                per_side += 5                    # whole joint is worth max-5
                already_counted.add(base)
        else:
            per_side += limits[s]

    return per_side * 2                          # L + R


# --------------------------------------------------------------------------- #
# 2)  generic “sum + extrapolate” – now forwards the new switch
# --------------------------------------------------------------------------- #
def sum_and_extrapolate_scores_df(
    df: pd.DataFrame,
    scores_to_sum: List[str],
    collapse_ED_EP_pairs: bool = False,
    fraction_required_valid_scores=0.0 
) -> pd.DataFrame:
    """
    • Sums labels & preds the usual way.
    • Scales the partial sums up to the full Sharp / van der Heijde domain
      (or whatever domain is implied by *scores_to_sum*).

    Set ``collapse_ED_EP_pairs=True`` **only** when you pass the complete
    hand-erosion score list that still contains both *ED* and *EP* entries.
    """
    df_summed = sum_scores_df(df, scores_to_sum=scores_to_sum)

    m = max_possible_score(
        scores_to_sum,
        collapse_ED_EP_pairs=collapse_ED_EP_pairs
    )

    df_summed["preds_summed_extrapolated"]  = df_summed["preds_summed"]  * m / df_summed["limit_summed"]
    df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * m / df_summed["limit_summed"]
    m_valid = (df_summed["limit_summed"] / m) >= fraction_required_valid_scores
    df_summed = df_summed[m_valid]  
    return df_summed


def sum_and_extrapolate_scores_df_JSN_H(df: pd.DataFrame, fraction_required_valid_scores=0.0): 
    df_summed = sum_scores_df_JSN_H(df)
    m = 120
    df_summed["preds_summed_extrapolated"]  = df_summed["preds_summed"]  * m / df_summed["limit_summed"]
    df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * m / df_summed["limit_summed"]
    m_valid = (df_summed["limit_summed"] / m) >= fraction_required_valid_scores
    df_summed = df_summed[m_valid]    
    return df_summed


def sum_and_extrapolate_scores_df_JSN_F(df: pd.DataFrame, fraction_required_valid_scores=0.75): 
    df_summed = sum_scores_df_JSN_F(df)
    m = 48
    df_summed["preds_summed_extrapolated"]  = df_summed["preds_summed"]  * m / df_summed["limit_summed"]
    df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * m / df_summed["limit_summed"]
    m_valid = (df_summed["limit_summed"] / m) >= fraction_required_valid_scores
    df_summed = df_summed[m_valid]    
    return df_summed


def sum_and_extrapolate_scores_df_ERO_F(df: pd.DataFrame,
                                        fraction_required_valid_scores=0.0):
    df_summed = sum_scores_df_ERO_F(df)
    m = 120
    df_summed["preds_summed_extrapolated"]  = df_summed["preds_summed"]  * m / df_summed["limit_summed"]
    df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * m / df_summed["limit_summed"]

    m_valid = (df_summed["limit_summed"] / m) >= fraction_required_valid_scores
    df_summed = df_summed[m_valid]    
    return df_summed



def sum_and_extrapolate_scores_df_ERO_H(
    df: pd.DataFrame,
    limit_treatment_ED: Literal["cap E_D sum to 5", "E_D_mean"] = "cap E_D sum to 5",
    fraction_required_valid_scores=0.0 # for 0 all are returned 
) -> pd.DataFrame:
    
    df_summed = sum_scores_df_ERO_H(df, limit_treatment_ED=limit_treatment_ED)
    max_total = 80 * 2   # left + right  → 160
    df_summed["preds_summed_extrapolated"]  = df_summed["preds_summed"]  * max_total / df_summed["limit_summed"]
    df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * max_total / df_summed["limit_summed"]

    m_valid = (df_summed["limit_summed"] / max_total) >= fraction_required_valid_scores
    df_summed = df_summed[m_valid]
    return df_summed




#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

def generate_score_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-patient-date table (output of `sum_and_extrapolate_scores_df`),
    produce every ordered pair of visits (dateA > dateB) for each patient and
    the deltas between their extrapolated sums.

    Output columns
    --------------
    patientId
    dateA, dateB                              – str, exactly as in patientId_date
    preds_summed_extrapolated_A   (float)
    preds_summed_extrapolated_B   (float)
    labels_summed_extrapolated_A  (float)
    labels_summed_extrapolated_B  (float)
    labels_summed_extrapolated_delta  (float) – B minus A
    preds_summed_extrapolated_delta   (float) – B minus A
    """

    # --- prepare -----------------------------------------------------------------
    work = df.copy()

    # split patientId_date into its components
    work["patientId"] = work["patientId_date"].str.split("_").str[0]
    work["date"]      = pd.to_datetime(work["patientId_date"].str.split("_").str[1])

    # keep only the columns we actually need in the merge
    small = work[
        ["patientId",
         "date",
         "preds_summed_extrapolated",
         "labels_summed_extrapolated"]
    ]

    # --- self-merge to get every pair within the same patient --------------------
    pairs = (
        small.merge(small,
                    on="patientId",
                    suffixes=("_A", "_B"))
              # keep only (dateA > dateB)  ── i.e. dateA is the later visit
              .query("date_A < date_B")
    )

    # --- compute deltas ----------------------------------------------------------
    pairs["labels_summed_extrapolated_delta"] = (
        pairs["labels_summed_extrapolated_B"] -
        pairs["labels_summed_extrapolated_A"]
    )
    pairs["preds_summed_extrapolated_delta"] = (
        pairs["preds_summed_extrapolated_B"] -
        pairs["preds_summed_extrapolated_A"]
    )

    # --- final tidy-up -----------------------------------------------------------
    out = (
        pairs.rename(columns={
                "date_A": "dateA",
                "date_B": "dateB"})
             .assign(
                # convert dates back to plain strings so the column types
                # match your original code’s expectations
                dateA = lambda d: d["dateA"].dt.strftime("%Y-%m-%d"),
                dateB = lambda d: d["dateB"].dt.strftime("%Y-%m-%d"),
             )
             # reorder columns to the spec
             [["patientId",
               "dateA", "dateB",
               "preds_summed_extrapolated_A", "preds_summed_extrapolated_B",
               "labels_summed_extrapolated_A", "labels_summed_extrapolated_B",
               "labels_summed_extrapolated_delta", "preds_summed_extrapolated_delta"]]
             .reset_index(drop=True)
    )

    return out


def add_extrapolation(df_summed: pd.DataFrame,
                      scores_to_sum: List[str]) -> pd.DataFrame:
    m = max_possible_score(scores_to_sum)      # ← unchanged
    out = df_summed.copy()
    out["preds_summed_extrapolated"]  = out["preds_summed"]  * m / out["limit_summed"]
    out["labels_summed_extrapolated"] = out["labels_summed"] * m / out["limit_summed"]
    return out

#---------------------------------------------#
#---------------------------------------------#
# ---------------------------------------------------------------------------
# joint-level meta data  (exactly the same as yesterday, just extended by caps)
# ---------------------------------------------------------------------------
# proximal+distal erosion pairs  ………………………………………
ERO_H_PAIRS = {        # each entry: (P-name, D-name, cap)
    'MCPIE'   : ('MCPIEP',   'MCPIED',   5),
    'MCPIIE'  : ('MCPIIEP',  'MCPIIED',  5),
    'MCPIIIE' : ('MCPIIIEP', 'MCPIIIED', 5),
    'MCPIVE'  : ('MCPIVEP',  'MCPIVED',  5),
    'MCPVE'   : ('MCPVEP',   'MCPVED',   5),
    'IPIE'    : ('IPIEP',    'IPIED',    5),
    'PIPIIE'  : ('PIPIIEP',  'PIPIIED',  5),
    'PIPIIIE' : ('PIPIIIEP', 'PIPIIIED', 5),
    'PIPIVE'  : ('PIPIVEP',  'PIPIVED',  5),
    'PIPVE'   : ('PIPVEP',   'PIPVED',   5),
}
# single-site erosions (hand wrist etc.) ………………… cap = 5 each
ERO_H_SINGLE = {'Base_MCIE': 5, 'RadiusE': 5, 'UlnaE': 5,
                'ScaphE': 5, 'LunatE': 5, 'TrapE': 5}

# foot erosions ………………………………………………………………………
ERO_F_PAIRS = {
    'MTPIE'   : ('MTPIEP',   'MTPIED',   10),
    'MTPIIE'  : ('MTPIIEP',  'MTPIIED',  10),
    'MTPIIIE' : ('MTPIIIEP', 'MTPIIIED', 10),
    'MTPIVE'  : ('MTPIVEP',  'MTPIVED',  10),
    'MTPVE'   : ('MTPVEP',   'MTPVED',   10),
    'IPE'     : ('IPEP',     'IPED',     10),
}
ERO_F_SINGLE = {}          # none

# JSN – no proximal/distal halves ……………………………… cap = 4
JSN_H = {j: 4 for j in
         ['CMCIII', 'CMCIV', 'CMCV',
          'MCPI', 'MCPII', 'MCPIII', 'MCPIV', 'MCPV',
          'PIPII', 'PIPIII', 'PIPIV', 'PIPV',
          'Rad_Carp', 'Sca_Cap', 'Tra_Sca']}

JSN_F = {j: 4 for j in
         ['IP', 'MTPI', 'MTPII', 'MTPIIII', 'MTPIV', 'MTPV']}


# ---------------------------------------------------------------------------
# low-level helpers  (unchanged except that we now *return* limit_used)
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _pivot(df_long: pd.DataFrame) -> pd.DataFrame:
    """labels/preds wide table, duplicate joints summed."""
    return df_long.pivot_table(index='patientId_date',
                               columns='score_type',
                               values=['labels', 'preds'],
                               aggfunc='sum')

def _pair(row, p, d, cap) -> Tuple[float,float,bool,int]:
    la, lb = row[('labels', p)], row[('labels', d)]
    pa, pb = row[('preds',  p)], row[('preds',  d)]
    ok = not (np.isnan(la)|np.isnan(lb)|np.isnan(pa)|np.isnan(pb))
    if not ok: return np.nan, np.nan, False, 0
    return min(la+lb, cap), min(pa+pb, cap), True, cap

def _single(row, j, cap) -> Tuple[float,float,bool,int]:
    l, p = row[('labels', j)], row[('preds', j)]
    ok = not (np.isnan(l)|np.isnan(p))
    if not ok: return np.nan, np.nan, False, 0
    return l, p, True, cap


def _aggregate(df_long: pd.DataFrame,
               pairs: Dict[str, Tuple[str,str,int]],
               singles: Dict[str, int],
               is_ero: bool) -> pd.DataFrame:
    wide = _pivot(df_long)
    recs=[]
    for pid,row in wide.iterrows():
        lsum = psum = 0.0
        limit_used = n_valid = 0
        # pairs
        for _, (p,d,cap) in pairs.items():
            l,pred,ok,cap_used = _pair(row,p,d,cap)
            if ok:
                lsum+=l; psum+=pred; limit_used+=cap_used; n_valid+=1
        # singles
        for j,cap in singles.items():
            l,pred,ok,cap_used = _single(row,j,cap)
            if ok:
                lsum+=l; psum+=pred; limit_used+=cap_used; n_valid+=1
        recs.append(dict(patientId_date=pid,
                         labels_summed=lsum,
                         preds_summed=psum,
                         limit_summed=limit_used,
                         n_valid_ERO = n_valid if is_ero else 0,
                         n_valid_JSN = n_valid if not is_ero else 0))
    return pd.DataFrame(recs)


def sum_scores_ERO_H(df_long):
    return _aggregate(df_long, ERO_H_PAIRS, ERO_H_SINGLE, is_ero=True)

def sum_scores_JSN_H(df_long):
    return _aggregate(df_long, {}, JSN_H, is_ero=False)

def sum_scores_ERO_F(df_long):
    return _aggregate(df_long, ERO_F_PAIRS, ERO_F_SINGLE, is_ero=True)

def sum_scores_JSN_F(df_long):
    return _aggregate(df_long, {}, JSN_F, is_ero=False)

#######################################################################

# import numpy as np
# import pandas as pd
# from typing import List, Dict, Tuple

# # ────────────────────────────────────────────────────────────────────────────
# #  joint meta-data (foot erosions shown here; hand ERO / JSN tables analogous)
# # ────────────────────────────────────────────────────────────────────────────
# #           base-name : (P-row, D-row, cap)
# ERO_F_PAIRS: Dict[str, Tuple[str, str, int]] = {
#     "MTPIE"  : ("MTPIEP",   "MTPIED",   10),
#     "MTPIIE" : ("MTPIIEP",  "MTPIIED",  10),
#     "MTPIIIE": ("MTPIIIEP", "MTPIIIED", 10),
#     "MTPIVE" : ("MTPIVEP",  "MTPIVED",  10),
#     "MTPVE"  : ("MTPVEP",   "MTPVED",   10),
#     "IPE"    : ("IPEP",     "IPED",     10),
# }
# ERO_F_SINGLE: Dict[str, int] = {}        # none in the feet

# # helper: *all* score_type names we need for this sub-score ------------
# ERO_F_SCORE_NAMES: List[str] = (
#       [n for tup in ERO_F_PAIRS.values() for n in tup[:2]]   # P & D
#     + list(ERO_F_SINGLE.keys())
# )


# # ────────────────────────────────────────────────────────────────────────────
# #  the aggregation routine
# # ────────────────────────────────────────────────────────────────────────────
# def sum_scores_ERO_F(df_long: pd.DataFrame) -> pd.DataFrame:
#     """
#     Parameters
#     ----------
#     df_long  – the original joint table with columns
#                (patientId_date, score_type, JSN_or_ERO, labels, preds, …)

#     Returns the classical “summed” table with
#         patientId_date, labels_summed, preds_summed,
#         limit_summed,  n_valid_ERO, n_valid_JSN
#     where limit_summed is the *row-wise* cap sum – identical to the
#     legacy   sum_scores_df → sum_and_extrapolate_scores_df   behaviour.
#     """

#     # 0) retain only the score rows belonging to this sub-score
#     sel = df_long[df_long["score_type"].isin(ERO_F_SCORE_NAMES)].copy()

#     # ------------------------------------------------------------------
#     # 1) limit_summed – row-wise cap sum (left + right handled naturally)
#     # ------------------------------------------------------------------
#     cap_lookup = {n: cap                       for n_cap in
#                   ([(*tup,)] for tup in ERO_F_PAIRS.values())}
#     # easier: build once
#     cap_lookup = {p:cap for _,(p,d,cap) in ERO_F_PAIRS.items()} | \
#                  {d:cap for _,(p,d,cap) in ERO_F_PAIRS.items()} | \
#                  ERO_F_SINGLE

#     sel["limit_row"] = sel["score_type"].map(cap_lookup)
#     limit_summed = (sel
#                     .groupby("patientId_date")["limit_row"]
#                     .sum()
#                     .rename("limit_summed"))

#     # ------------------------------------------------------------------
#     # 2) counts of *valid* rows (labels & preds both non-NaN)
#     # ------------------------------------------------------------------
#     valid_mask = (~sel["labels"].isna()) & (~sel["preds"].isna())
#     n_valid = (sel[valid_mask]
#                .groupby("patientId_date")
#                .size()
#                .rename("n_valid_ERO"))

#     # ------------------------------------------------------------------
#     # 3) labels_summed & preds_summed
#     #    – need the proximal+distal clipping logic
#     # ------------------------------------------------------------------
#     # pivot wide to have every score_type once per patient-date
#     wide = sel.pivot_table(index="patientId_date",
#                            columns="score_type",
#                            values=["labels", "preds"],
#                            aggfunc="sum")

#     out_rows = []
#     for pid, row in wide.iterrows():
#         L = P = 0.0

#         # P/D pairs  …………………
#         for name, (p_name, d_name, cap) in ERO_F_PAIRS.items():
#             l_p, l_d = row[("labels", p_name)], row[("labels", d_name)]
#             p_p, p_d = row[("preds",  p_name)], row[("preds",  d_name)]
#             if not np.isnan(l_p) and not np.isnan(l_d):
#                 L += min(l_p + l_d, cap)
#             if not np.isnan(p_p) and not np.isnan(p_d):
#                 P += min(p_p + p_d, cap)

#         # single joints (none for foot erosions, but code kept generic)
#         for j, cap in ERO_F_SINGLE.items():
#             l, p = row[("labels", j)], row[("preds", j)]
#             if not np.isnan(l): L += l
#             if not np.isnan(p): P += p

#         out_rows.append({"patientId_date": pid,
#                          "labels_summed": L,
#                          "preds_summed":  P})

#     sums = pd.DataFrame(out_rows).set_index("patientId_date")

#     # ------------------------------------------------------------------
#     # 4) final tidy-up – add limit & valid counts, zero-fill missing ids
#     # ------------------------------------------------------------------
#     out = (sums
#            .join(limit_summed, how="outer")
#            .join(n_valid,       how="outer")
#            .fillna({"n_valid_ERO": 0,
#                     "labels_summed": 0,
#                     "preds_summed":  0})
#            .assign(n_valid_JSN = 0)          # required downstream
#            .reset_index())

#     return out