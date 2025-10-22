#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_impute_states.py
-------------------
Cleans the raw patient-states table produced in step 3:

 • removes / clamps outliers
 • infers FiO₂, GCS, BP constituents, temperatures, Hb/Ht, bilirubin
 • sample-and-hold fills short gaps
 • (optional) creates a provenance log and/or a ±1/0 change-mask file
"""

from __future__ import annotations

import argparse
import os
from typing import Final

import numpy as np
import pandas as pd
from tqdm import tqdm

from AI_clinician.preprocessing.columns import *         # noqa: F403
from AI_clinician.preprocessing.imputation import (
    fill_outliers,
    fill_stepwise,
    sample_and_hold,
)
from AI_clinician.preprocessing.provenance import ProvenanceWriter
from AI_clinician.preprocessing.utils import load_csv

# --------------------------------------------------------------------- #
#  Paths                                                                #
# --------------------------------------------------------------------- #
PARENT_DIR: Final[str] = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #
def handle_weight_measurements(df: pd.DataFrame, provenance: ProvenanceWriter | None):
    """
    Handle weight measurements by:
    1. Detecting physiologically impossible changes (>5% change within 4 hours)
    2. Keeping the most reliable measurement when multiple exist
    """
    # Sort by icustay_id and timestep
    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP])
    
    # Calculate time differences in hours
    df['time_diff'] = df.groupby(C_ICUSTAYID)[C_TIMESTEP].diff() / 3600
    
    # Calculate weight changes
    df['weight_diff'] = df.groupby(C_ICUSTAYID)[C_WEIGHT].diff()
    df['weight_pct_change'] = (df['weight_diff'] / df.groupby(C_ICUSTAYID)[C_WEIGHT].shift()) * 100
    
    # Find impossible changes (>5% within 4 hours)
    impossible_changes = (abs(df['weight_pct_change']) > 5) & (df['time_diff'] <= 4)
    
    if provenance and impossible_changes.any():
        provenance.record(
            "Impossible weight change detected",
            row=df.index[impossible_changes],
            col=C_WEIGHT
        )
    
    # For each impossible change, keep the measurement that's closer to the median weight
    for icu in df[C_ICUSTAYID].unique():
        icu_mask = df[C_ICUSTAYID] == icu
        icu_impossible = impossible_changes & icu_mask
        
        if icu_impossible.any():
            # Get the median weight for this ICU stay
            median_weight = df.loc[icu_mask, C_WEIGHT].median()
            
            # For each impossible change, keep the measurement closer to median
            for idx in df.index[icu_impossible]:
                prev_idx = df.index[df.index.get_loc(idx) - 1]
                if abs(df.loc[idx, C_WEIGHT] - median_weight) > abs(df.loc[prev_idx, C_WEIGHT] - median_weight):
                    df.loc[idx, C_WEIGHT] = df.loc[prev_idx, C_WEIGHT]
    
    # Clean up temporary columns
    df = df.drop(['time_diff', 'weight_diff', 'weight_pct_change'], axis=1)
    return df

def remove_outliers(df: pd.DataFrame, provenance: ProvenanceWriter | None):
    """Clamp obviously wrong values & log provenance."""
    wrong_unit_t = (df[C_TEMP_C] > 90) & df[C_TEMP_F].isna()
    if provenance:
        provenance.record(
            "temp_F logged as temp_C",
            row=df.index[wrong_unit_t],
            col=C_TEMP_F,
            reference_col=C_TEMP_C,
        )
    df.loc[wrong_unit_t, C_TEMP_F] = df.loc[wrong_unit_t, C_TEMP_C]

    small_fio2 = df[C_FIO2_100] < 1
    if provenance:
        provenance.record(
            "FiO2_100 < 1",
            row=df.index[small_fio2],
            col=C_FIO2_100,
        )
    df.loc[small_fio2, C_FIO2_100] *= 100

    # Handle weight measurements before other outlier removal
    df = handle_weight_measurements(df, provenance)

    df = fill_outliers(
        df,
        {
            # ──Vitals───────────────────────────────────────────────
            C_WEIGHT: (None, 300),
            C_HR: (None, 250),
            C_SYSBP: (0, 300),
            C_MEANBP: (0, 200),
            C_DIABP: (0, 200),
            C_RR: (None, 80),
            C_SPO2: (None, 150),
            C_TEMP_C: (None, 90),
            C_FIO2_100: (20, None),
            C_FIO2_1: (None, 1.5),
            C_O2FLOW: (None, 70),
            C_PEEP: (0, 40),
            C_TIDALVOLUME: (None, 1800),
            C_MINUTEVENTIL: (None, 50),
            # ──Labs────────────────────────────────────────────────
            C_POTASSIUM: (1, 15),
            C_SODIUM: (95, 178),
            C_CHLORIDE: (70, 150),
            C_GLUCOSE: (1, 1000),
            C_CREATININE: (None, 150),
            C_MAGNESIUM: (None, 10),
            C_CALCIUM: (None, 20),
            C_IONISED_CA: (None, 5),
            C_CO2_MEQL: (None, 120),
            C_SGPT: (None, 10000),
            C_SGOT: (None, 10000),
            C_HB: (None, 20),
            C_HT: (None, 65),
            C_WBC_COUNT: (None, 500),
            C_PLATELETS_COUNT: (None, 2000),
            C_INR: (None, 20),
            C_ARTERIAL_PH: (6.7, 8),
            C_PAO2: (None, 700),
            C_PACO2: (None, 200),
            C_ARTERIAL_BE: (-50, None),
            C_ARTERIAL_LACTATE: (None, 30),
        },
        provenance=provenance,
    )

    # hard-cap SpO₂ at 100 %
    big_spo2 = df[C_SPO2] > 100
    if provenance:
        provenance.record(
            "Clamp SpO2 > 100", row=df.index[big_spo2], col=C_SPO2
        )
    df.loc[big_spo2, C_SPO2] = 100
    return df


def convert_fio2_units(
    df: pd.DataFrame,
    provenance: ProvenanceWriter | None = None,
    note: str | None = None,
):
    """Synchronise FiO₂_{100,1} representations."""
    need_set = df[C_FIO2_1].isna() & df[C_FIO2_100].notna()
    if provenance and need_set.any():
        provenance.record(
            "FiO2_1 <- FiO2_100",
            row=df.index[need_set],
            col=C_FIO2_1,
            reference_col=C_FIO2_100,
            metadata=note,
        )
    df.loc[need_set, C_FIO2_1] = df.loc[need_set, C_FIO2_100] / 100

    need_pct = df[C_FIO2_1].notna() & df[C_FIO2_100].isna()
    if provenance and need_pct.any():
        provenance.record(
            "FiO2_100 <- FiO2_1",
            row=df.index[need_pct],
            col=C_FIO2_100,
            reference_col=C_FIO2_1,
            metadata=note,
        )
    df.loc[need_pct, C_FIO2_100] = df.loc[need_pct, C_FIO2_1] * 100
    return df


def estimate_fio2(df: pd.DataFrame, provenance: ProvenanceWriter | None):
    """Heuristic FiO₂ inference (unchanged from reference)."""
    df = convert_fio2_units(df, provenance, "R1")

    sah = {}
    for col in (C_INTERFACE, C_FIO2_100, C_O2FLOW):
        print(f"SAH on {col}")
        sah[col] = sample_and_hold(
            df[C_ICUSTAYID], df[C_TIMESTEP], df[col], SAH_HOLD_DURATION[col]
        )

    # --- rules C1 … C6 (verbatim) --------------------------------------
    def apply_rule(mask, vals, tag):
        if provenance and mask.any():
            provenance.record(
                "FiO2 estimation", row=df.index[mask], col=C_FIO2_100, metadata=tag
            )
        df.loc[mask, C_FIO2_100] = vals

    # C1
    mask = (
        sah[C_FIO2_100].isna()
        & sah[C_O2FLOW].notna()
        & sah[C_INTERFACE].isin((0, 2))
    )
    apply_rule(mask, fill_stepwise(sah[C_O2FLOW][mask], zip(
        [15, 12, 10, 8, 6, 5, 4, 3, 2, 1],
        [70, 62, 55, 50, 44, 40, 36, 32, 28, 24],
    )), "C1")

    # C2
    mask = (
        sah[C_FIO2_100].isna()
        & sah[C_O2FLOW].isna()
        & sah[C_INTERFACE].isin((0, 2))
    )
    apply_rule(mask, 21, "C2")

    # C3
    mask = (
        sah[C_FIO2_100].isna()
        & sah[C_O2FLOW].notna()
        & (
            sah[C_INTERFACE].isna()
            | sah[C_INTERFACE].isin((1, 3, 4, 5, 6, 9, 10))
        )
    )
    apply_rule(mask, fill_stepwise(sah[C_O2FLOW][mask], zip(
        [15, 12, 10, 8, 6, 4],
        [75, 69, 66, 58, 40, 36],
    )), "C3")

    # C4
    mask = (
        sah[C_FIO2_100].isna()
        & sah[C_O2FLOW].isna()
        & (
            sah[C_INTERFACE].isna()
            | sah[C_INTERFACE].isin((1, 3, 4, 5, 6, 9, 10))
        )
    )
    apply_rule(mask, pd.NA, "C4")

    # C5
    mask = (
        sah[C_FIO2_100].isna()
        & sah[C_O2FLOW].notna()
        & (sah[C_INTERFACE] == 7)
    )
    apply_rule(mask, fill_stepwise(
        sah[C_O2FLOW][mask],
        zip([9.99, 8, 6], [80, 70, 60]),
        zip([10, 15], [90, 100]),
    ), "C5")

    # C6
    mask = (
        sah[C_FIO2_100].isna()
        & sah[C_O2FLOW].isna()
        & (sah[C_INTERFACE] == 7)
    )
    apply_rule(mask, pd.NA, "C6")

    df = convert_fio2_units(df, provenance, "R2")
    return df


def estimate_gcs(rass):
    """Richmond → Glasgow conversion (returns float / np.nan)."""
    if pd.isna(rass):
        return np.nan
    rass = float(rass)
    if rass >= 0:
        return 15.0
    return { -1: 14, -2: 12, -3: 11, -4: 6, -5: 3 }.get(rass, np.nan)


def estimate_vitals(df: pd.DataFrame, provenance: ProvenanceWriter | None):
    """BP-triad, temperatures, Hb/Ht, bilirubin."""
    # ──BP────────────
    print("BP ", end="")
    m = df[C_SYSBP].notna() & df[C_MEANBP].notna() & df[C_DIABP].isna()
    if provenance and m.any():
        provenance.record("BP estimation", row=df.index[m], col=C_DIABP)
    df.loc[m, C_DIABP] = (3 * df.loc[m, C_MEANBP] - df.loc[m, C_SYSBP]) / 2

    m = df[C_SYSBP].notna() & df[C_DIABP].notna() & df[C_MEANBP].isna()
    if provenance and m.any():
        provenance.record("BP estimation", row=df.index[m], col=C_MEANBP)
    df.loc[m, C_MEANBP] = (
        df.loc[m, C_SYSBP] + 2 * df.loc[m, C_DIABP]
    ) / 3

    m = df[C_MEANBP].notna() & df[C_DIABP].notna() & df[C_SYSBP].isna()
    if provenance and m.any():
        provenance.record("BP estimation", row=df.index[m], col=C_SYSBP)
    df.loc[m, C_SYSBP] = 3 * df.loc[m, C_MEANBP] - 2 * df.loc[m, C_DIABP]
    print("[DONE]")

    # ──Temperatures──
    print("TEMP ", end="")
    tf2c = (df[C_TEMP_F] > 25) & (df[C_TEMP_F] < 45)
    if provenance and tf2c.any():
        provenance.record("Temp_C <- Temp_F", row=df.index[tf2c], col=C_TEMP_C)
    df.loc[tf2c, C_TEMP_C] = df.loc[tf2c, C_TEMP_F]
    df.loc[tf2c, C_TEMP_F] = np.nan

    tc2f = df[C_TEMP_C] > 70
    if provenance and tc2f.any():
        provenance.record("Temp_F <- Temp_C", row=df.index[tc2f], col=C_TEMP_F)
    df.loc[tc2f, C_TEMP_F] = df.loc[tc2f, C_TEMP_C]
    df.loc[tc2f, C_TEMP_C] = np.nan

    m = df[C_TEMP_C].notna() & df[C_TEMP_F].isna()
    if provenance and m.any():
        provenance.record("Temp_F from Temp_C", row=df.index[m], col=C_TEMP_F)
    df.loc[m, C_TEMP_F] = df.loc[m, C_TEMP_C] * 1.8 + 32

    m = df[C_TEMP_F].notna() & df[C_TEMP_C].isna()
    if provenance and m.any():
        provenance.record("Temp_C from Temp_F", row=df.index[m], col=C_TEMP_C)
    df.loc[m, C_TEMP_C] = (df.loc[m, C_TEMP_F] - 32) / 1.8
    print("[DONE]")

    # ──Hb / Ht───────
    print("Hb/Ht ", end="")
    m = df[C_HB].notna() & df[C_HT].isna()
    if provenance and m.any():
        provenance.record("Ht from Hb", row=df.index[m], col=C_HT)
    df.loc[m, C_HT] = df.loc[m, C_HB] * 2.862 + 1.216

    m = df[C_HT].notna() & df[C_HB].isna()
    if provenance and m.any():
        provenance.record("Hb from Ht", row=df.index[m], col=C_HB)
    df.loc[m, C_HB] = (df.loc[m, C_HT] - 1.216) / 2.862
    print("[DONE]")

    # ──Bilirubin────
    print("BILI ", end="")
    m = df[C_TOTAL_BILI].notna() & df[C_DIRECT_BILI].isna()
    if provenance and m.any():
        provenance.record("Direct bili from total", row=df.index[m], col=C_DIRECT_BILI)
    df.loc[m, C_DIRECT_BILI] = df.loc[m, C_TOTAL_BILI] * 0.6934 - 0.1752

    m = df[C_DIRECT_BILI].notna() & df[C_TOTAL_BILI].isna()
    if provenance and m.any():
        provenance.record("Total bili from direct", row=df.index[m], col=C_TOTAL_BILI)
    df.loc[m, C_TOTAL_BILI] = (df.loc[m, C_DIRECT_BILI] + 0.1752) / 0.6934
    print("[DONE]")

    return df


def safe_change_mask(old: pd.Series, new: pd.Series) -> np.ndarray:
    """1 = added/changed, −1 = removed, 0 = unchanged (handles NA safely)."""
    added = old.isna() & new.notna()
    removed = old.notna() & new.isna()
    changed = (
        old.notna()
        & new.notna()
        & old.fillna(np.nan).ne(new.fillna(np.nan))
    )
    out = np.zeros(len(old), dtype=np.int8)
    out[added | changed] = 1
    out[removed] = -1
    return out


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean / impute the patient-state table (step 4)."
    )
    parser.add_argument("input", help="patient_states.csv from step 3")
    parser.add_argument("output", help="destination for cleaned CSV")
    parser.add_argument(
        "--data",
        dest="data_dir",
        help="root data dir (defaults to ../data)",
        default=None,
    )
    parser.add_argument(
        "--no-outliers",
        dest="do_outliers",
        action="store_false",
        default=True,
        help="skip outlier removal",
    )
    parser.add_argument(
        "--no-fio2",
        dest="do_fio2",
        action="store_false",
        default=True,
        help="skip FiO2 estimation",
    )
    parser.add_argument(
        "--no-gcs",
        dest="do_gcs",
        action="store_false",
        default=True,
        help="skip GCS estimation",
    )
    parser.add_argument(
        "--no-vitals",
        dest="do_vitals",
        action="store_false",
        default=True,
        help="skip other vital/lab inference",
    )
    parser.add_argument(
        "--no-sample-and-hold",
        dest="do_sah",
        action="store_false",
        default=True,
        help="skip sample-and-hold filling",
    )
    parser.add_argument(
        "--mask-file",
        dest="mask_file",
        help="write ±1/0 mask of changes compared with input",
    )
    parser.add_argument(
        "--provenance-dir",
        dest="prov_dir",
        help="write detailed provenance records to this dir",
    )

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(PARENT_DIR, "data")

    df = load_csv(args.input)
    original = df.copy(deep=False) if args.mask_file else None

    prov = (
        ProvenanceWriter(args.prov_dir, verbose=True) if args.prov_dir else None
    )

    if args.do_outliers:
        print("Removing outliers …")
        df = remove_outliers(df, provenance=prov)

    if args.do_gcs:
        print("Estimating GCS …")
        need = df[C_GCS].isna()
        if prov and need.any():
            prov.record(
                "Estimate GCS from RASS",
                row=df.index[need],
                col=C_GCS,
                reference_col=C_RASS,
            )
        gcs_est = pd.to_numeric(
            df.loc[need, C_RASS].apply(estimate_gcs), errors="coerce"
        )
        df.loc[need, C_GCS] = gcs_est

    if args.do_fio2:
        print("Estimating FiO₂ …")
        df = estimate_fio2(df, provenance=prov)

    if args.do_vitals:
        print("Estimating vitals …")
        df = estimate_vitals(df, provenance=prov)

    if args.do_sah:
        print("Running sample-and-hold …")
        out = {
            C_BLOC: df[C_BLOC],
            C_ICUSTAYID: df[C_ICUSTAYID],
            C_TIMESTEP: df[C_TIMESTEP],
        }
        for col in SAH_FIELD_NAMES:
            print(f"SAH on {col}")
            out[col] = sample_and_hold(
                df[C_ICUSTAYID],
                df[C_TIMESTEP],
                df[col],
                SAH_HOLD_DURATION[col],
                provenance=prov,
                col_name=col,
            )
        df = pd.DataFrame(out)

    # ------------------------------------------------------------------
    print("Writing cleaned CSV …")
    df.to_csv(args.output, index=False, float_format="%g")
    if prov:
        prov.close()

    # ------------------------------------------------------------------
    if args.mask_file:
        print("Writing mask file …")
        mask_cols = {
            C_BLOC: df[C_BLOC],
            C_ICUSTAYID: df[C_ICUSTAYID],
            C_TIMESTEP: df[C_TIMESTEP],
        }
        shared = set(original.columns).intersection(df.columns) - set(mask_cols)
        for col in shared:
            mask_cols[col] = safe_change_mask(original[col], df[col])
        pd.DataFrame(mask_cols).to_csv(args.mask_file, index=False)