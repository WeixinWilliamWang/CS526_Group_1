#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_preprocess_sepsis_onset.py
-----------------------------
Compute first moment of suspected infection for each ICU stay
and write data/intermediates/sepsis_onset.csv
"""

import argparse
import glob
import os

import pandas as pd
from tqdm import tqdm

from AI_clinician.preprocessing.columns import (
    C_ICUSTAYID, C_CHARTTIME, C_STARTDATE
)
from AI_clinician.preprocessing.utils import (
    load_csv, load_intermediate_or_raw_csv
)
from AI_clinician.preprocessing.derived_features import calculate_onset

# ─────────────────────────────────────────────────────────────────── #
#  Helpers                                                            #
# ─────────────────────────────────────────────────────────────────── #
def _to_datetime_seconds(col: pd.Series) -> pd.Series:
    """Convert numeric seconds to datetime, treating 0 / '', ' ' as NaT."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    col = col.replace(["", " ", 0, "0"], pd.NA)
    return pd.to_datetime(pd.to_numeric(col, errors="coerce"), unit="s", errors="coerce")


# Project root
PARENT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    )
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate sepsis onset time and write sepsis_onset.csv"
    )
    parser.add_argument("--data", dest="data_dir", default=None,
                        help="Base data directory (default = ../data/)")
    parser.add_argument("--out", dest="output_dir", default=None,
                        help="Output directory (default = data/intermediates)")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(PARENT_DIR, "data")
    out_dir  = args.output_dir or os.path.join(data_dir, "intermediates")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Antibiotics -------------------------------------------------- #
    abx = load_intermediate_or_raw_csv(data_dir, "abx.csv")
    abx[C_STARTDATE] = _to_datetime_seconds(abx[C_STARTDATE])

    # 2) Bacteriology ------------------------------------------------- #
    bacterio = load_csv(os.path.join(out_dir, "bacterio.csv"))
    bacterio[C_CHARTTIME] = _to_datetime_seconds(bacterio[C_CHARTTIME])

    # 3) Chart events (CE) ------------------------------------------- #
    ce_files = glob.glob(os.path.join(out_dir, "ce*.csv"))
    chart_events = pd.concat((load_csv(p) for p in ce_files), ignore_index=True)
    chart_events[C_CHARTTIME] = _to_datetime_seconds(chart_events[C_CHARTTIME])

    # 4) Lab events --------------------------------------------------- #
    lab_events = pd.concat(
        (
            load_intermediate_or_raw_csv(data_dir, fname)
            for fname in ("labs_ce.csv", "labs_le.csv")
        ),
        ignore_index=True,
    )
    lab_events[C_CHARTTIME] = _to_datetime_seconds(lab_events[C_CHARTTIME])

    # 5) Calculate onset per ICU stay -------------------------------- #
    records = []
    for stay_id in tqdm(
        abx[C_ICUSTAYID].dropna().unique(),
        desc="Calculating sepsis onset"
    ):
        onset = calculate_onset(
            abx,
            bacterio,
            stay_id,
            chart_events=chart_events,
            lab_events=lab_events
        )
        if onset is not None:
            records.append({
                C_ICUSTAYID: stay_id,
                "onset_time": int(onset.timestamp())    # keep seconds, consistent with 01
            })

    # 6) Save --------------------------------------------------------- #
    pd.DataFrame(records).to_csv(
        os.path.join(out_dir, "sepsis_onset.csv"),
        index=False
    )
    print(f"✓ Wrote {len(records):,} rows to sepsis_onset.csv")