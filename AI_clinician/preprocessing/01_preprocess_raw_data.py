#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_preprocess_raw_data.py
-------------------------

• Bucket-maps itemids in CE / LAB files.
• Cleans / normalises several raw tables.
• Builds bacterio.csv and imputes missing ICU-stay IDs.
• **Keeps every timestamp column (admittime, dischtime, intime, outtime,
  dob, dod, charttime, …) as an Int64 Unix-seconds value.**
"""

from __future__ import annotations

import argparse
import os
from typing import Final

import numpy as np
import pandas as pd
from tqdm import tqdm

from AI_clinician.preprocessing.columns import *                  # noqa: F403
from AI_clinician.preprocessing.imputation import impute_icustay_ids
from AI_clinician.preprocessing.utils import load_csv

tqdm.pandas()

# ------------------------------------------------------------------ #
#  Paths                                                             #
# ------------------------------------------------------------------ #
ROOT_DIR: Final[str] = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
RAW_DIR: Final[str] = os.path.join(ROOT_DIR, "data", "raw_data")
INT_DIR: Final[str] = os.path.join(ROOT_DIR, "data", "intermediates")

SECONDS_PER_DAY: Final[int] = 86_400
BAD_LAB_SENTINELS: Final[list[int | float]] = [
    999999, 999999.0, 99999, 99999.0, -999999, -999999.0
]

# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #
def _as_int_seconds(col: pd.Series) -> pd.Series:
    """
    Return *any* timestamp column as Int64 Unix-seconds.

    Treat the common "missing" sentinels (0, '', ' ')
    as <NA> so they don't interfere with arithmetic.
    """
    col = col.replace(["", " ", 0, "0"], pd.NA)
    return pd.to_numeric(col, errors="coerce").astype("Int64")


def _demog_seconds(demog: pd.DataFrame) -> pd.DataFrame:
    """Copy with intime / outtime as numeric seconds (needed by imputer)."""
    d = demog.copy()
    for c in (C_INTIME, C_OUTTIME):             # noqa: F405
        d[c] = _as_int_seconds(d[c])
    return d


# ------------------------------------------------------------------ #
#  Main routine                                                      #
# ------------------------------------------------------------------ #
def preprocess_raw_data(
    input_dir: str,
    output_dir: str,
    simplify_events: bool = True,
    no_bacterio: bool = False,
    no_gender_imputation: bool = False,
    no_elixhauser_zerofill: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("Preprocessing raw data…")

    # ─────────────────────────────────────────────────────────────── #
    # 1.  Bucket-map CE / LAB events                                  #
    # ─────────────────────────────────────────────────────────────── #
    if simplify_events:
        print("  → Simplifying chart (ce*) and lab (labs*) events")
        event_files = [
            fn for fn in os.listdir(input_dir)
            if fn.startswith(("ce", "labs")) and fn.endswith(".csv")
        ]
        for fn in tqdm(event_files):
            ref = REF_VITALS if fn.startswith("ce") else REF_LABS      # noqa: F405
            mapping = {
                item: bucket_idx + 1
                for bucket_idx, bucket in enumerate(ref)
                for item in bucket
            }

            df = pd.read_csv(
                os.path.join(input_dir, fn),
                dtype={C_ITEMID: "int32"}                              # noqa: F405
            )
            df[C_ITEMID] = df[C_ITEMID].replace(mapping)               # noqa: F405
            df.to_csv(os.path.join(output_dir, fn), index=False)

    # ─────────────────────────────────────────────────────────────── #
    # 2.  Demographics                                                #
    # ─────────────────────────────────────────────────────────────── #
    demog = load_csv(os.path.join(input_dir, "demog.csv"), null_icustayid=True)

    # convert *all* timestamp columns to Int64 seconds
    for col in (
        C_ADMITTIME, C_DISCHTIME, C_INTIME, C_OUTTIME, C_DOD, C_DOB   # noqa: F405
    ):
        demog[col] = _as_int_seconds(demog[col])

    # gender 1/2 → 1/0
    demog[C_GENDER] = demog[C_GENDER].map({1: 1, 2: 0})               # noqa: F405
    if not no_gender_imputation:
        demog[C_GENDER] = demog[C_GENDER].fillna(demog[C_GENDER].mode(dropna=True)[0])

    # --- mortality flags from seconds ---
    # Calculate 90-day mortality from ICU admission time
    diff_sec = demog[C_DOD].astype("float64") - demog[C_INTIME].astype("float64")  # noqa: F405
    diff_days = diff_sec / SECONDS_PER_DAY
    demog[C_MORTA_90] = ((0 <= diff_days) & (diff_days <= 90)).astype("Int64")        # noqa: F405
    demog[C_MORTA_HOSP] = (
        (demog[C_DOD] <= demog[C_DISCHTIME]) & demog[C_DOD].notna()                   # noqa: F405
    ).astype("Int64")

    # ─────────────────────────────────────────────────────────────── #
    # 3.  Labs  (optional latest-value columns)                       #
    # ─────────────────────────────────────────────────────────────── #
    if all(os.path.exists(os.path.join(input_dir, f)) for f in ("labs_ce.csv", "labs_le.csv")):
        print("  → Pulling latest lab values (Total_prot, CRP, ACT, ETCO2, SvO2)")
        labs = pd.concat([
            load_csv(os.path.join(input_dir, "labs_ce.csv")),
            load_csv(os.path.join(input_dir, "labs_le.csv")),
        ])

        LABS = {
            "Total_prot": [227429, 851, 51002, 51003],
            "CRP":        [227444, 50889],
            "ACT":        [779, 490, 3785, 3838, 3837, 50821,
                           220224, 226063, 226770, 227039],
            "ETCO2":      [225668, 1531, 50813],
            "SvO2":       [227443, 50882, 50803],
        }
        for name, ids in LABS.items():
            sub = labs[labs[C_ITEMID].isin(ids)]                     # noqa: F405
            sub = sub[~sub[C_VALUENUM].isin(BAD_LAB_SENTINELS)]      # noqa: F405
            sub = (
                sub.groupby([C_ICUSTAYID, C_CHARTTIME], as_index=False)[C_VALUENUM]  # noqa: F405
                .mean()
            )
            sub["__dt"] = pd.to_datetime(
                sub[C_CHARTTIME], unit="s", errors="coerce"
            )
            latest = sub.sort_values("__dt").groupby(C_ICUSTAYID).tail(1)           # noqa: F405
            demog = demog.merge(
                latest[[C_ICUSTAYID, C_VALUENUM]].rename(columns={C_VALUENUM: name}),  # noqa: F405
                how="left",
                on=C_ICUSTAYID
            )

    # ─────────────────────────────────────────────────────────────── #
    # 4.  Fluid_mv  (normalised infusion rate)                        #
    # ─────────────────────────────────────────────────────────────── #
    if os.path.exists(os.path.join(input_dir, "fluid_mv.csv")):
        print("  → Computing normalised infusion rate [fluid_mv]")
        fm = load_csv(os.path.join(input_dir, "fluid_mv.csv"), null_icustayid=True)
        fm[C_NORM_INFUSION_RATE] = fm[C_TEV] * fm[C_RATE] / fm[C_AMOUNT]             # noqa: F405
        fm.to_csv(os.path.join(output_dir, "fluid_mv.csv"), index=False)

    # ─────────────────────────────────────────────────────────────── #
    # 5.  Clean mechvent / vaso files                                 #
    # ─────────────────────────────────────────────────────────────── #
    def _clean(fname: str) -> None:
        src = os.path.join(input_dir, fname)
        if not os.path.exists(src):
            return
        df = load_csv(src, null_icustayid=True)
        df = df[df[C_ICUSTAYID].notna()]                                           # noqa: F405
        df.to_csv(os.path.join(output_dir, fname), index=False)

    _clean("mechvent.csv")
    _clean("vaso_mv.csv")
    _clean("vaso_cv.csv")
    _clean("fluid_cv.csv")

    # ─────────────────────────────────────────────────────────────── #
    # 6.  Antibiotics                                                 #
    # ─────────────────────────────────────────────────────────────── #
    print("  → Trimming unusable entries [abx]")
    abx = load_csv(os.path.join(input_dir, "abx.csv"), null_icustayid=True)
    abx = abx[abx[C_STARTDATE].notna() & abx[C_ICUSTAYID].notna()]                 # noqa: F405

    need_icu = abx[C_ICUSTAYID].isna()                                            # noqa: F405
    if need_icu.any():
        abx.loc[need_icu, C_ICUSTAYID] = impute_icustay_ids(
            _demog_seconds(demog), abx[need_icu]
        )
    abx.to_csv(os.path.join(output_dir, "abx.csv"), index=False)

    # ─────────────────────────────────────────────────────────────── #
    # 7.  Bacteriology                                                #
    # ─────────────────────────────────────────────────────────────── #
    if not no_bacterio:
        print("  → Building bacterio.csv")
        culture  = load_csv(os.path.join(input_dir, "culture.csv"),  null_icustayid=True)
        microbio = load_csv(os.path.join(input_dir, "microbio.csv"), null_icustayid=True)

        missing_ct = microbio[C_CHARTTIME].isna()                                 # noqa: F405
        microbio.loc[missing_ct, C_CHARTTIME] = microbio.loc[missing_ct, C_CHARTDATE]  # noqa: F405
        microbio[C_CHARTDATE] = 0                                                 # noqa: F405

        cols = [C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_CHARTTIME]                # noqa: F405
        bact = pd.concat([microbio[cols], culture[cols]], ignore_index=True).drop_duplicates()

        bact[C_CHARTTIME] = _as_int_seconds(bact[C_CHARTTIME])

        need_icu = bact[C_ICUSTAYID].isna()                                       # noqa: F405
        if need_icu.any():
            print(f"    Imputing {need_icu.sum():,} missing ICUStayIDs…")
            bact.loc[need_icu, C_ICUSTAYID] = impute_icustay_ids(
                _demog_seconds(demog), bact[need_icu]
            )

        bact = bact[bact[C_ICUSTAYID].notna()]                                    # noqa: F405
        bact.to_csv(os.path.join(output_dir, "bacterio.csv"), index=False)

    # ─────────────────────────────────────────────────────────────── #
    # 8.  Final demog save                                            #
    # ─────────────────────────────────────────────────────────────── #
    mortality_cols = [C_MORTA_90, C_MORTA_HOSP]
    if not no_elixhauser_zerofill:
        mortality_cols.append(C_ELIXHAUSER)
    
    for col in mortality_cols:                                                    # noqa: F405
        demog[col] = demog[col].fillna(0)
    demog.to_csv(os.path.join(output_dir, "demog.csv"), index=False)

    print("✔ Preprocessing complete.")


# ------------------------------------------------------------------ #
#  CLI wrapper                                                       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw MIMIC-IV CSVs.")
    parser.add_argument("--in",  dest="input_dir",  default=RAW_DIR,
                        help="Raw data directory (default: data/raw_data/)")
    parser.add_argument("--out", dest="output_dir", default=INT_DIR,
                        help="Output directory (default: data/intermediates/)")
    parser.add_argument("--no-events",   dest="no_events",   action="store_true",
                        help="Skip CE / LAB simplification")
    parser.add_argument("--no-bacterio", dest="no_bacterio", action="store_true",
                        help="Skip building bacterio.csv")
    parser.add_argument("--no-gender-imputation", dest="no_gender_imputation", action="store_true",
                        help="Skip gender mode imputation for missing values")
    parser.add_argument("--no-elixhauser-zerofill", dest="no_elixhauser_zerofill", action="store_true",
                        help="Skip Elixhauser zero-fill imputation for missing values")
    args = parser.parse_args()

    preprocess_raw_data(
        args.input_dir,
        args.output_dir,
        simplify_events=not args.no_events,
        no_bacterio=args.no_bacterio,
        no_gender_imputation=args.no_gender_imputation,
        no_elixhauser_zerofill=args.no_elixhauser_zerofill,
    )
