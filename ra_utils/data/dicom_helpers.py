from pathlib import Path
import pandas as pd


import monai
from monai.data import Dataset, CacheDataset, DataLoader

import monai
from monai.data import PILReader
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage, Spacingd, SpatialCropd, ResizeWithPadOrCropd


import numpy as np
import os
import yaml

from tqdm import tqdm
import re
from typing import Tuple

#from ra_utils.data.crawler_helpers import scantree, get_and_maybe_save_crawler_data




# ------------------------------------------------------------
#  dicom_helpers.py
# ------------------------------------------------------------


##############################################################################
# Dictionaries / keyword banks
##############################################################################

_HAND_KWS = [
    "hand", "hände", "haende", "hande" #, "handgelenk", "wrist", "finger", "fingerkuppe",
]


_FOOT_KWS = [
    "foot", "feet", "fuß", "füße", "fuss", "fuesse", "fusse"
]

_VORFOOT_KWS = [
    "vorfuß", "vorfuss", "forefoot"
]
_KNEE_KWS = [
    "knee", "knie", "genu", "genuvalgum", "genuflexum", "genuvarum"
]


_LEFT_KWS  = ["l", "left", "sin", "links", "lk", "li"]
_RIGHT_KWS = ["r", "right", "dex", "rechts", "rk", "re"]


import math, re

def _norm(txt: str) -> str:
    """
    lower-case, strip accents, collapse whitespace.
    """
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = txt.replace("ß", "ss").replace("ẞ", "ss")
    return re.sub(r"\s+", " ", txt.lower()).strip()


# ------------------------------------------------------------------
# 1) universal “safe-string” helper  (replaces the old `_lower`)
# ------------------------------------------------------------------
def _s(val: object) -> str:
    """Return lowercase string; empty if None/NaN."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).lower()

def safe_str(val: object) -> str:
    """Return lowercase string; empty if None/NaN."""
    return _s(val)

# ------------------------------------------------------------------
# 2) use _s(...) everywhere, never add raw values
# ------------------------------------------------------------------
def laterality(row) -> str:
    lat = _s(row.get("Laterality"))
    if lat in _LEFT_KWS:
        return "L"
    if lat in _RIGHT_KWS:
        return "R"

    desc = f"{_s(row.get('SeriesDescription'))} {_s(row.get('StudyDescription'))}"
    desc = _norm(desc)
    if any(k in desc.split() for k in _LEFT_KWS):
        return "L"
    if any(k in desc.split() for k in _RIGHT_KWS):
        return "R"
    return "NA"

def body_part(row) -> str:
    bp = _s(row.get("BodyPartExamined"))
    if any(k in bp for k in _HAND_KWS):
        return "H"
    if any(k in bp for k in _FOOT_KWS):
        return "F"

    desc = " ".join(_s(row.get(c)) for c in
                    ["SeriesDescription", "StudyDescription", "CodeMeaning"])
    desc = _norm(desc)
    if any(k in desc for k in _HAND_KWS):
        return "H"
    if any(k in desc for k in _FOOT_KWS):
        return "F"
    if any(k in desc for k in _VORFOOT_KWS):
        return "VF"
    if any(k in desc for k in _KNEE_KWS):
        return "K"
    
    return "NA"

# add all common oblique/lateral codes + German words ⬇︎
_VIEW_KWS = {
    #  ↙︎--- keep what you had ---↘︎
    "pa": "pa", 
    "ap": "ap", 
    "dp": "dp", 
    "dpa": "dp",
    "dorsoplantar": "dp",

    # ---- oblique (→ 'ob') ------------------------------
    "oblique": "ob", 
    "obl": "ob", 
    "ob": "ob",
    "rao": "ob", "lao": "ob",             # R/L anterior oblique
    "rpo": "ob", "lpo": "ob",             # R/L posterior oblique
    "rlo": "ob", "llo": "ob",             # R/L lateral oblique
    "mlo": "ob",                          # mammo: mediolateral oblique
    "schräg": "ob", 
    "schrag": "ob",       # German
    "schraeg": "ob",       # German

    # ---- lateral (→ 'lat') -----------------------------
    "lat": "lat", 
    "lateral": "lat",
    "ll": "lat",  
    "rl": "lat",            # left/right lateral
    "profil": "lat", 
    "seitlich": "lat",   # German
    "seitl.": "lat",  # German

    # ---- axial (→ 'ax') -------------------------------
    "ax": "ax", "axial": "ax",
}
import unicodedata, re





def view(row) -> str:
    # ---------- 1) direct ViewPosition ----------
    vp = _s(row.get("ViewPosition"))
    if vp and (abbr := _VIEW_KWS.get(_norm(vp))):
        return abbr

    # ---------- 2) free text fallback ----------
    desc = " ".join(_s(row.get(c)) for c in
                    ["SeriesDescription", "StudyDescription", "CodeMeaning"])
    desc_norm = _norm(desc)

    # longest keys first so ‘oblique’ matches before ‘ob’
    for kw, abbr in sorted(_VIEW_KWS.items(), key=lambda kv: -len(kv[0])):
        if kw in desc_norm.split() or kw in desc_norm:
            return abbr

    return "NA"

def clean_patient_name(name):
    name = _s(name).split("^")[0].split()[0]
    name = re.sub(r"\W+", "", name)
    return name or "unknown"

def img_id(row):
    pid  = clean_patient_name(row.get("PatientName"))
    date = _s(row.get("StudyDate")) or _s(row.get("ContentDate"))
    bp   = body_part(row)
    lat  = laterality(row)
    vp   = view(row)
    return f"{pid}_{date}_{bp}_{lat}_{vp}".rstrip("_")

def add_img_id_column(df):
    df["img_id"] = df.apply(img_id, axis=1)
    return df
