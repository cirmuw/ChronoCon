import pandas as pd
import pingouin as pg


def compute_icc3(df_wide, score_name, is_aggregated):
    df_wide = df_wide.copy()
    """Return one-row dataframe with ICC(3,1) for a single score."""
    c_origin = f"{score_name}__origin"
    c_D      = f"{score_name}__D"
    
    assert (df_wide["patientId_visitDate__origin"] == df_wide["patientId_visitDate__D"]).all()
    assert (df_wide["bodypart_manual__origin"] == df_wide["bodypart_manual__D"]).all()
    

    cols = ["patientId_visitDate__origin", "bodypart_manual__origin", c_origin, c_D]

    # TODO try to convert to int. If fails NA
    # coerce non-numeric into NaN, then convert to nullable Int64 dtype
    df_wide[c_D]      = pd.to_numeric(df_wide[c_D],      errors="coerce").astype("Int64")
    df_wide[c_origin] = pd.to_numeric(df_wide[c_origin], errors="coerce").astype("Int64")
    df = df_wide[cols].dropna(subset=[c_origin, c_D]).copy()
    

    # build a proper 'target' column
    if is_aggregated:
        # keep exactly one row per visit ID (drop duplicate L/R rows)
        df = df.drop_duplicates(subset=["patientId_visitDate__origin"])
        df["target"] = df["patientId_visitDate__origin"].astype(str)
    else:
        # joint-level → visitID_side (e.g. 1234-2024-01-01_L)
        df["target"] = (
            df["patientId_visitDate__origin"].astype(str) + "_" +
            df["bodypart_manual__origin"].astype(str)
        )

    # convert from wide → long
    df_long = pd.melt(
        df,
        id_vars    = ["target"],
        value_vars = [c_origin, c_D],
        var_name   = "Raters",
        value_name = "Rating"
    )

    # trim suffix so we just have 'origin' / 'D'
    df_long["Raters"] = df_long["Raters"].str.split("__").str[-1]

    # ICC
    icc_all = pg.intraclass_corr(
        data    = df_long,
        targets = "target",
        raters  = "Raters",
        ratings = "Rating"
    ).round(3)

    icc3 = icc_all.loc[icc_all["Type"] == "ICC3"].copy()
    icc3["score_name"] = score_name
    return icc3