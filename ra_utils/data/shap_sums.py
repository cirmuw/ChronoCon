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
    
    # Check if any of the dataframes are empty before concatenation
    if len(df_summed_pairs) == 0 and len(df_summed_singles) == 0:
        # Return an empty dataframe with the expected columns
        return pd.DataFrame(columns=["patientId_date", "labels_summed", "preds_summed", "limit_summed"])
    elif len(df_summed_pairs) == 0:
        df_all = df_summed_singles.copy()
    elif len(df_summed_singles) == 0:
        df_all = df_summed_pairs.copy()
    else:
        # Ensure consistent dtypes before concatenation to avoid FutureWarning
        columns_to_align = ['patientId_date', 'labels_summed', 'preds_summed', 'limit_summed']
        for col in columns_to_align:
            if col in df_summed_pairs.columns and col in df_summed_singles.columns:
                dtype = max(df_summed_pairs[col].dtype, df_summed_singles[col].dtype)
                df_summed_pairs[col] = df_summed_pairs[col].astype(dtype)
                df_summed_singles[col] = df_summed_singles[col].astype(dtype)
        
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


def sum_and_extrapolate_scores_df_JSN_H(df: pd.DataFrame, fraction_required_valid_scores=0.0, max_total = 120): 
    df_summed = sum_scores_df_JSN_H(df)
    m = max_total
    df_summed["preds_summed_extrapolated"]  = df_summed["preds_summed"]  * m / df_summed["limit_summed"]
    df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * m / df_summed["limit_summed"]
    m_valid = (df_summed["limit_summed"] / m) >= fraction_required_valid_scores
    df_summed = df_summed[m_valid]    
    return df_summed


def sum_and_extrapolate_scores_df_JSN_F(df: pd.DataFrame, fraction_required_valid_scores=0.75, max_total = 48): 
    df_summed = sum_scores_df_JSN_F(df)
    m = max_total
    df_summed["preds_summed_extrapolated"]  = df_summed["preds_summed"]  * m / df_summed["limit_summed"]
    df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * m / df_summed["limit_summed"]
    m_valid = (df_summed["limit_summed"] / m) >= fraction_required_valid_scores
    df_summed = df_summed[m_valid]    
    return df_summed


def sum_and_extrapolate_scores_df_ERO_F(df: pd.DataFrame,
                                        fraction_required_valid_scores=0.0, 
                                        max_total = 120):
    df_summed = sum_scores_df_ERO_F(df)
    m = max_total
    df_summed["preds_summed_extrapolated"]  = df_summed["preds_summed"]  * m / df_summed["limit_summed"]
    df_summed["labels_summed_extrapolated"] = df_summed["labels_summed"] * m / df_summed["limit_summed"]

    m_valid = (df_summed["limit_summed"] / m) >= fraction_required_valid_scores
    df_summed = df_summed[m_valid]    
    return df_summed



def sum_and_extrapolate_scores_df_ERO_H(
    df: pd.DataFrame,
    limit_treatment_ED: Literal["cap E_D sum to 5", "E_D_mean"] = "cap E_D sum to 5",
    fraction_required_valid_scores=0.0, # for 0 all are returned 
    max_total = 160
) -> pd.DataFrame:
    
    df_summed = sum_scores_df_ERO_H(df, limit_treatment_ED=limit_treatment_ED)
    # max_total = 80 * 2   # left + right  → 160
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


