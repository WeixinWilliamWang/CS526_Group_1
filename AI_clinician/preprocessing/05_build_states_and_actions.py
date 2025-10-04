#!/usr/bin/env python
import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from ai_clinician.preprocessing.columns import *
from ai_clinician.preprocessing.utils import load_csv, load_intermediate_or_raw_csv

PARENT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    )
)

def build_states_and_actions(df, qstime,
                             inputMV, inputCV, inputpreadm,
                             vasoMV, vasoCV,
                             demog,
                             UOpreadm, UO,
                             timestep_resolution,
                             winb4, winaft,
                             head=None, allowed_stays=None):
    """
    Bins each patient's time‐series into intervals of length `timestep_resolution` hours
    within an 80h window around sepsis onset, and appends fluid, vasopressor, UO,
    demographic/outcome and computed metrics (shock index, P/F ratio).
    """
    # get all stays, filter & head
    stays = sorted(df[C_ICUSTAYID].dropna().unique())
    if allowed_stays is not None:
        stays = [s for s in stays if s in set(allowed_stays)]
    if head:
        stays = stays[:head]
    print(f"{len(stays)} ICU stay IDs")

    combined = []
    # pre-allocate mapping of raw→binned rows
    bin_ix = pd.Series(index=df.index, dtype=pd.Int64Dtype())

    # total_duration = (winb4+3)+(winaft+3) = 80 hours
    total_duration = (winb4 + 3) + (winaft + 3)

    for sid in tqdm(stays, desc="Building states & actions"):
        # 1) subset chart/lab
        patient = df[df[C_ICUSTAYID] == sid]
        if patient.empty:
            continue
        beg = int(patient[C_TIMESTEP].iloc[0])

        # 2) fluids (MetaVision + optional CareVue)
        mv = inputMV[inputMV[C_ICUSTAYID] == sid]
        # coerce
        start_mv = pd.to_numeric(mv[C_STARTTIME], errors="coerce")
        end_mv   = pd.to_numeric(mv[C_ENDTIME],   errors="coerce")
        rate_mv  = mv.get(C_NORM_INFUSION_RATE,
                          mv.get(C_RATE, pd.Series(np.nan, index=mv.index)))
        # optional CV
        if inputCV is not None:
            cv = inputCV[inputCV[C_ICUSTAYID] == sid]
            ct_cv = pd.to_numeric(cv[C_CHARTTIME], errors="coerce")
        else:
            cv, ct_cv = None, None

        # 3) vasopressors
        vaso1 = vasoMV[vasoMV[C_ICUSTAYID] == sid]
        st_v = pd.to_numeric(vaso1[C_STARTTIME], errors="coerce")
        en_v = pd.to_numeric(vaso1[C_ENDTIME],   errors="coerce")
        rv   = vaso1[C_RATESTD]
        if vasoCV is not None:
            vaso2 = vasoCV[vasoCV[C_ICUSTAYID] == sid]
            ct_v2 = pd.to_numeric(vaso2[C_CHARTTIME], errors="coerce")
        else:
            vaso2, ct_v2 = None, None

        # 4) demographics & outcomes
        d = demog[demog[C_ICUSTAYID] == sid].iloc[0]
        dod    = pd.to_numeric(d[C_DOD], errors="coerce")
        outt   = pd.to_numeric(d[C_OUTTIME], errors="coerce")
        discht = pd.to_numeric(qstime.loc[sid, C_DISCHTIME], errors="coerce")
        lastt  = pd.to_numeric(qstime.loc[sid, C_LAST_TIMESTEP], errors="coerce")
        gender = pd.to_numeric(d[C_GENDER], errors="coerce") or 0
        mort_h = pd.to_numeric(d[C_MORTA_HOSP], errors="coerce") or 0
        mort_90= pd.to_numeric(d[C_MORTA_90], errors="coerce")  or 0

        dem = {
            C_GENDER: gender,
            C_AGE:     d[C_AGE],
            C_ELIXHAUSER: d[C_ELIXHAUSER],
            C_RE_ADMISSION: (d[C_ADM_ORDER] > 1),
            C_DIED_IN_HOSP: mort_h,
            C_DIED_WITHIN_48H_OF_OUT_TIME:
                (not np.isnan(dod) and not np.isnan(outt) and abs(dod-outt) < 2*24*3600),
            C_MORTA_90: mort_90,
            C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH:
                ((discht - lastt)/3600) if not np.isnan(discht) and not np.isnan(lastt) else np.nan
        }

        # 5) pre‐window fluid volume
        pre_m = inputpreadm[inputpreadm[C_ICUSTAYID] == sid][C_INPUT_PREADM].sum()
        t0_pre, t1_pre = 0, beg
        # infusion before window
        inf_pre = np.nansum( rate_mv * (end_mv-start_mv) *
                             ((end_mv<=t1_pre)&(start_mv>=t0_pre)) / 3600 +
                             rate_mv * (end_mv-t0_pre) *
                             ((start_mv<=t0_pre)&(end_mv>=t0_pre)&(end_mv<=t1_pre)) / 3600 +
                             rate_mv * (t1_pre-start_mv) *
                             ((start_mv>=t0_pre)&(start_mv<=t1_pre)&(end_mv>=t1_pre)) / 3600 +
                             rate_mv * (t1_pre-t0_pre) *
                             ((start_mv<=t0_pre)&(end_mv>=t1_pre)) / 3600 )
        # boluses before window (MV)
        bol_pre_mv = mv.loc[
            mv[C_RATE].isna() &
            (start_mv>=t0_pre)&(start_mv<=t1_pre),
            C_TEV
        ].sum()
        # bolus CV
        if cv is not None:
            bol_pre_cv = cv.loc[(ct_cv>=t0_pre)&(ct_cv<=t1_pre), C_TEV].sum()
        else:
            bol_pre_cv = 0
        totvol_pre = pre_m + inf_pre + bol_pre_mv + bol_pre_cv

        # 6) pre‐window UO
        pre_u = UOpreadm[UOpreadm[C_ICUSTAYID] == sid][C_VALUE].sum()
        outp = UO[UO[C_ICUSTAYID] == sid]
        ct_u = pd.to_numeric(outp[C_CHARTTIME], errors="coerce")
        uo_pre = outp.loc[(ct_u>=t0_pre)&(ct_u<=t1_pre), C_VALUE].sum()
        UOtot_pre = pre_u + uo_pre

        # now step through each bin
        totvol = totvol_pre
        UOtot  = UOtot_pre

        for j in np.arange(0, total_duration, timestep_resolution):
            t0 = beg + int(3600*j)
            t1 = beg + int(3600*(j + timestep_resolution))

            # collect vitals/labs in [t0,t1]
            vals = patient[(patient[C_TIMESTEP]>=t0)&(patient[C_TIMESTEP]<=t1)]
            if vals.empty:
                continue

            row = {
                C_BLOC: (j//timestep_resolution) + 1,
                C_ICUSTAYID: sid,
                C_TIMESTEP: int(t0),
                **dem
            }

            # a) vitals & labs: mean
            for col in SAH_FIELD_NAMES:
                row[col] = vals[col].mean(skipna=True)

            # b) vasopressor
            mask1 = ((en_v>=t0)&(en_v<=t1)) | ((st_v>=t0)&(en_v<=t1)) \
                  | ((st_v>=t0)&(st_v<=t1)) | ((st_v<=t0)&(en_v>=t1))
            vs1 = rv.loc[mask1].values if not vaso1.empty else np.array([])
            if vaso2 is not None:
                vs2 = vaso2.loc[(ct_v2>=t0)&(ct_v2<=t1), C_RATESTD].values
            else:
                vs2 = np.array([])
            all_vs = np.concatenate([vs1, vs2])
            all_vs = all_vs[~np.isnan(all_vs)]
            row[C_MEDIAN_DOSE_VASO] = float(np.nanmedian(all_vs)) if all_vs.size else 0.0
            row[C_MAX_DOSE_VASO]    = float(np.nanmax(all_vs))    if all_vs.size else 0.0

            # c) fluid this bin
            infu = np.nansum( rate_mv * (end_mv-start_mv) *
                              ((end_mv<=t1)&(start_mv>=t0)) / 3600 +
                              rate_mv * (end_mv-t0) *
                              ((start_mv<=t0)&(end_mv>=t0)&(end_mv<=t1)) / 3600 +
                              rate_mv * (t1-start_mv) *
                              ((start_mv>=t0)&(start_mv<=t1)&(end_mv>=t1)) / 3600 +
                              rate_mv * (t1-t0) *
                              ((start_mv<=t0)&(end_mv>=t1)) / 3600 )
            bol_mv = mv.loc[(mv[C_RATE].isna()) & (start_mv>=t0)&(start_mv<=t1), C_TEV].sum()
            if cv is not None:
                bol_cv = cv.loc[(ct_cv>=t0)&(ct_cv<=t1), C_TEV].sum()
            else:
                bol_cv = 0
            step_vol = infu + bol_mv + bol_cv
            totvol += step_vol
            row[C_INPUT_STEP]  = step_vol
            row[C_INPUT_TOTAL] = totvol

            # d) UO this bin
            uo_now = outp.loc[(ct_u>=t0)&(ct_u<=t1), C_VALUE].sum()
            UOtot += uo_now
            row[C_OUTPUT_STEP]  = uo_now
            row[C_OUTPUT_TOTAL] = UOtot

            # Calculate cumulative fluid balance
            row[C_CUMULATED_BALANCE] = totvol - UOtot

            # e) computed: shock index and P/F ratio
            hr   = row.get(C_HR,   np.nan)
            sbp  = row.get(C_SYSBP,np.nan)
            pao2 = row.get(C_PAO2, np.nan)
            fio1 = row.get(C_FIO2_1, np.nan)
            # shock index
            row[C_SHOCK_INDEX] = hr/sbp if (not np.isnan(hr) and not np.isnan(sbp)) else np.nan
            # PaO2/FiO2 ratio
            row[C_PAO2_FIO2]   = pao2/fio1 if (not np.isnan(pao2) and not np.isnan(fio1) and fio1>0) else np.nan

            # record mapping
            bin_ix[vals.index] = len(combined)
            combined.append(row)

    states = pd.DataFrame(combined)
    # ensure every expected column exists
    expected = (DEMOGRAPHICS_FIELD_NAMES +
                SAH_FIELD_NAMES +
                IO_FIELD_NAMES +
                COMPUTED_FIELD_NAMES)
    for c in expected:
        if c not in states:
            states[c] = pd.NA

    mapping = pd.DataFrame({
        C_BLOC:       df[C_BLOC],
        C_ICUSTAYID:  df[C_ICUSTAYID],
        C_TIMESTEP:   df[C_TIMESTEP],
        C_BIN_INDEX:  bin_ix
    })
    return states, mapping


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="05: Bin patient_states into fixed‐width windows and add fluids/UO/vaso/actions"
    )
    p.add_argument("input",   help="Raw patient_states.csv")
    p.add_argument("qstime",  help="qstime.csv (indexed by icustayid)")
    p.add_argument("output",  help="Where to write binned patient_states_filled.csv")
    p.add_argument("--mapping-file", dest="mapping_file", default=None,
                   help="Optional: write a row‐to‐bin mapping CSV")
    p.add_argument("--data", dest="data_dir", default=None,
                   help="Base data dir (default = ../data)")
    p.add_argument("--resolution", type=float, default=4.0,
                   help="Hours per bin (default=4h)")
    p.add_argument("--window-before", type=int, default=49,
                   help="Hours before onset (default=49)")
    p.add_argument("--window-after", type=int, default=25,
                   help="Hours after onset (default=25)")
    p.add_argument("--head", type=int, default=None,
                   help="Limit to first N ICU stays")
    p.add_argument("--filter-stays", dest="filter_stays_path", default=None,
                   help="CSV with icustayid to restrict")

    args = p.parse_args()
    data_dir = args.data_dir or os.path.join(PARENT_DIR, "data")

    # load inputs
    print("Loading inputs…")
    df      = load_csv(args.input)
    qstime  = load_csv(args.qstime).set_index(C_ICUSTAYID, drop=True)

    demog      = load_intermediate_or_raw_csv(data_dir, "demog.csv")
    inputpreadm= load_intermediate_or_raw_csv(data_dir, "preadm_fluid.csv")
    inputMV    = load_intermediate_or_raw_csv(data_dir, "fluid_mv.csv")
    vasoMV     = load_intermediate_or_raw_csv(data_dir, "vaso_mv.csv")
    # optional CV
    try:
        inputCV = load_intermediate_or_raw_csv(data_dir, "fluid_cv.csv")
    except FileNotFoundError:
        inputCV = None
    try:
        vasoCV  = load_intermediate_or_raw_csv(data_dir, "vaso_cv.csv")
    except FileNotFoundError:
        vasoCV  = None
    UOpreadm = load_intermediate_or_raw_csv(data_dir, "preadm_uo.csv")
    UO       = load_intermediate_or_raw_csv(data_dir, "uo.csv")

    allowed = None
    if args.filter_stays_path:
        allowed = load_csv(args.filter_stays_path)[C_ICUSTAYID]

    # build & write
    states, mapping = build_states_and_actions(
        df, qstime,
        inputMV, inputCV, inputpreadm,
        vasoMV, vasoCV,
        demog,
        UOpreadm, UO,
        args.resolution,
        args.window_before,
        args.window_after,
        head=args.head,
        allowed_stays=allowed
    )

    print(f"Writing binned states → {args.output}")
    states.to_csv(args.output, index=False, float_format="%g")

    if args.mapping_file:
        print(f"Writing mapping → {args.mapping_file}")
        mapping.to_csv(args.mapping_file, index=False, float_format="%g")