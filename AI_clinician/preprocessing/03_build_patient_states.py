#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_build_patient_states.py
--------------------------
• Takes the outputs of steps 01 & 02 and builds:

    └── patient_states.csv   one row per (icustayid, timestamp)
    └── qstime.csv           helper table with onset / window bounds

All timestamps remain Unix-seconds integers; no binning or imputation is
performed here.
"""

from __future__ import annotations

import argparse
import os
from typing import Final, List

import pandas as pd
from tqdm import tqdm
import numpy as np

from ai_clinician.preprocessing.columns import *
from ai_clinician.preprocessing.utils import (
    load_csv,
    load_intermediate_or_raw_csv,
)

# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #
ROOT_DIR: Final[str] = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)


def _to_int_seconds(col: pd.Series) -> pd.Series:
    """Return Int64 seconds; treat 0 / '' / ' ' as <NA>."""
    col = col.replace(["", " ", 0, "0"], pd.NA)
    return pd.to_numeric(col, errors="coerce").astype("Int64")


class ChartEvents:
    """Thin wrapper around a list of CE dataframes for quick per-stay lookup."""

    def __init__(self, dfs: List[pd.DataFrame], stay_id_col: str = C_ICUSTAYID):
        self.dfs = dfs
        self.stay_id_col = stay_id_col

    def fetch(self, stay_id: int) -> pd.DataFrame:
        parts = [
            df[df[self.stay_id_col] == stay_id] for df in self.dfs
        ]
        return (
            pd.concat(parts, ignore_index=True)
            if any(len(p) for p in parts)
            else pd.DataFrame(columns=self.dfs[0].columns)
        )


def time_window(
    df: pd.DataFrame,
    col: str,
    centre: int,
    lower: int,
    upper: int,
) -> pd.DataFrame:
    """Rows whose *col* ∈ [centre − lower, centre + upper] (all ints)."""
    if df.empty:
        return df
    ts = pd.to_numeric(df[col], errors="coerce")
    return df[(ts >= centre - lower) & (ts <= centre + upper)]


# ------------------------------------------------------------------ #
#  Core routine                                                      #
# ------------------------------------------------------------------ #
def build_patient_states(
    chart_events: ChartEvents,
    onset: pd.DataFrame,
    demog: pd.DataFrame,
    labU: pd.DataFrame,
    MV: pd.DataFrame,
    MV_proc: pd.DataFrame | None,
    win_before_h: int,
    win_after_h: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined: list[dict] = []
    qstime_records: list[dict] = []
    proc_hits = 0

    lower_s = (win_before_h + 4) * 3600   # seconds
    upper_s = (win_after_h + 4) * 3600

    for _, row in tqdm(onset.iterrows(), total=len(onset), desc="Building states"):
        icu = int(row[C_ICUSTAYID])
        onset_sec = int(row[C_ONSET_TIME])
        if onset_sec <= 0:
            continue

        age_discharge = demog.loc[demog[C_ICUSTAYID] == icu,
                                  [C_AGE, C_DISCHTIME]].values
        if len(age_discharge) == 0 or age_discharge[0, 0] < 18:
            continue
        disch_time = int(age_discharge[0, 1])

        # pull windows
        ce_win = time_window(chart_events.fetch(icu), C_CHARTTIME, onset_sec,
                             lower_s, upper_s)
        lab_win = time_window(labU[labU[C_ICUSTAYID] == icu], C_CHARTTIME,
                              onset_sec, lower_s, upper_s)
        mv_win = time_window(MV[MV[C_ICUSTAYID] == icu], C_CHARTTIME,
                             onset_sec, lower_s, upper_s)
        if MV_proc is not None:
            mvp_win = time_window(MV_proc[MV_proc[C_ICUSTAYID] == icu],
                                  C_STARTTIME, onset_sec, lower_s, upper_s)
        else:
            mvp_win = pd.DataFrame(columns=[C_STARTTIME, C_MECHVENT, C_EXTUBATED])

        # union of all timestamps
        ts_union = sorted(pd.unique(pd.concat([
            ce_win[C_CHARTTIME], lab_win[C_CHARTTIME], mv_win[C_CHARTTIME],
            mvp_win[C_STARTTIME] if not mvp_win.empty else pd.Series([], dtype="Int64")
        ], ignore_index=True)))

        if not ts_union:
            continue

        for bloc, ts in enumerate(ts_union):
            row_dict: dict = {
                C_BLOC: bloc,
                C_ICUSTAYID: icu,
                C_TIMESTEP: int(ts)
            }

            # vitals
            for _, ev in ce_win[ce_win[C_CHARTTIME] == ts].iterrows():
                idx = ev[C_ITEMID] - 1
                if 0 <= idx < len(CHART_FIELD_NAMES):
                    colname = CHART_FIELD_NAMES[idx]
                    value = ev[C_VALUENUM]
                    
                    # Special handling for height only - convert inches to cm for adults
                    if colname == C_HEIGHT:
                        # Convert inches to cm if value is clearly in inches (< 100 for adults)
                        if value < 100:
                            value = value * 2.54  # inches to cm conversion
                        
                        # If we already have a height value, apply same conversion logic
                        if colname in row_dict and not pd.isna(row_dict[colname]):
                            current_val = row_dict[colname] 
                            if current_val < 100:
                                current_val = current_val * 2.54
                            # Keep the more recent (converted) value
                            row_dict[colname] = value
                        else:
                            row_dict[colname] = value
                    else:
                        # All other vitals: use original overwriting behavior
                        row_dict[colname] = value

            # labs
            for _, ev in lab_win[lab_win[C_CHARTTIME] == ts].iterrows():
                idx = ev[C_ITEMID] - 1
                if 0 <= idx < len(LAB_FIELD_NAMES):
                    row_dict[LAB_FIELD_NAMES[idx]] = ev[C_VALUENUM]

            # mechvent (chart)
            mv_slice = mv_win[mv_win[C_CHARTTIME] == ts]
            if not mv_slice.empty:
                ev = mv_slice.iloc[0]
                row_dict[C_MECHVENT] = ev[C_MECHVENT]
                row_dict[C_EXTUBATED] = ev[C_EXTUBATED]

            # mechvent (procedure)
            if not mvp_win.empty and (mvp_win[C_STARTTIME] == ts).any():
                subset = mvp_win[mvp_win[C_STARTTIME] == ts]
                row_dict[C_MECHVENT] = int(subset[C_MECHVENT].any())
                row_dict[C_EXTUBATED] = int(subset[C_EXTUBATED].any())
                proc_hits += 1

            combined.append(row_dict)

        qstime_records.append({
            C_ICUSTAYID:      icu,
            C_ONSET_TIME:     onset_sec,
            C_FIRST_TIMESTEP: int(ts_union[0]),
            C_LAST_TIMESTEP:  int(ts_union[-1]),
            C_DISCHTIME:      disch_time,
        })

    print(f"✔  Used {proc_hits:,} MV flags from procedure events")

    state_df = pd.DataFrame(combined)
    qstime_df = (
        pd.DataFrame(qstime_records)
        .set_index(C_ICUSTAYID)
        .astype("Int64")
    )

    # add any missing chart / lab / vent columns
    expected = CHART_FIELD_NAMES + LAB_FIELD_NAMES + VENT_FIELD_NAMES
    for col in expected:
        if col not in state_df.columns:
            state_df[col] = pd.NA

    return state_df, qstime_df


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3 – build patient_states (no binning, no imputation)."
    )
    parser.add_argument(
        "output_dir",
        help="Folder for patient_states.csv (e.g. data/intermediates/patient_states)"
    )
    parser.add_argument("--data", dest="data_dir", default=None,
                        help="Base data folder (raw + intermediates)")
    parser.add_argument("--window-before", type=int, default=49,
                        help="Hours before onset to include (default 49)")
    parser.add_argument("--window-after", type=int, default=25,
                        help="Hours after onset to include (default 25)")
    parser.add_argument("--head", type=int, default=None,
                        help="Keep only first N rows of onset data (dev/debug)")
    parser.add_argument("--filter-stays", dest="filter_stays_path",
                        help="CSV with icustayid column to restrict analysis")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(ROOT_DIR, "data")
    out_dir  = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── load chartevents (raw or simplified) ─────────────────────── #
    print("Reading chartevents …")
    raw_ce  = [f for f in os.listdir(os.path.join(data_dir, "raw_data"))
               if f.startswith("ce") and f.endswith(".csv")]
    int_ce  = [f for f in os.listdir(os.path.join(data_dir, "intermediates"))
               if f.startswith("ce") and f.endswith(".csv")]
    ce_dfs = [load_intermediate_or_raw_csv(data_dir, fname) for fname in raw_ce + int_ce]
    for df in ce_dfs:
        df[C_CHARTTIME] = _to_int_seconds(df[C_CHARTTIME])
    chart_events = ChartEvents(ce_dfs)

    # ── onset & demographics ─────────────────────────────────────── #
    onset = load_csv(os.path.join(data_dir, "intermediates", "sepsis_onset.csv"))
    demog = load_intermediate_or_raw_csv(data_dir, "demog.csv")

    # apply optional filters
    if args.filter_stays_path:
        allowed = load_csv(args.filter_stays_path)[C_ICUSTAYID]
        onset = onset[onset[C_ICUSTAYID].isin(allowed)]
    if args.head:
        onset = onset.head(args.head)

    # ── labs ─────────────────────────────────────────────────────── #
    print("Reading labs …")
    labU = pd.concat([
        load_intermediate_or_raw_csv(data_dir, "labs_ce.csv"),
        load_intermediate_or_raw_csv(data_dir, "labs_le.csv")
    ], ignore_index=True)
    labU[C_CHARTTIME] = _to_int_seconds(labU[C_CHARTTIME])

    # ── mechvent ─────────────────────────────────────────────────── #
    print("Reading mechvent …")
    MV = load_intermediate_or_raw_csv(data_dir, "mechvent.csv")
    MV[C_CHARTTIME] = _to_int_seconds(MV[C_CHARTTIME])
    try:
        MV_proc = load_intermediate_or_raw_csv(data_dir, "mechvent_pe.csv")
        MV_proc[C_STARTTIME] = _to_int_seconds(MV_proc[C_STARTTIME])
    except FileNotFoundError:
        MV_proc = None

    # ── build ────────────────────────────────────────────────────── #
    states, qstime = build_patient_states(
        chart_events,
        onset,
        demog,
        labU,
        MV,
        MV_proc,
        args.window_before,
        args.window_after
    )

    print(f"Result: {len(states):,} rows  ×  {len(states.columns)} columns")
    states.to_csv(os.path.join(out_dir, "patient_states.csv"),
                  index=False, float_format="%g")
    qstime.to_csv(os.path.join(out_dir, "qstime.csv"),
                  float_format="%g")
    print("✓ patient_states.csv and qstime.csv written.")