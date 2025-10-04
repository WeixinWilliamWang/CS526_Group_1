import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------- utils ----------------------------
def safe_cols(df, candidates):
    return [c for c in candidates if c in df.columns]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------- plotting basics ----------------------------
def plot_hist(df, col, outdir, bins=40, tag=None):
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    plt.figure()
    plt.hist(s, bins=bins)
    title = f"{col} Distribution" + (f" ({tag})" if tag else "")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    fname = os.path.join(outdir, f"hist_{col}" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_bar_counts(df, col, outdir, tag=None):
    if col not in df.columns:
        return None
    vc = df[col].value_counts(dropna=False).sort_index()
    if vc.empty:
        return None
    plt.figure()
    x = vc.index.astype(str)
    y = vc.values
    plt.bar(x, y)
    title = f"{col} Counts" + (f" ({tag})" if tag else "")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    fname = os.path.join(outdir, f"bar_{col}" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_box_by_gender(df, col, outdir, gender_col="gender", tag=None):
    if gender_col not in df.columns or col not in df.columns:
        return None
    sub = df[[col, gender_col]].dropna()
    if sub.empty or sub[gender_col].nunique() < 2:
        return None
    groups = [sub[sub[gender_col] == g][col].values for g in sorted(sub[gender_col].unique())]
    labels = [str(int(g)) if isinstance(g, (int, float)) and float(g).is_integer() else str(g)
              for g in sorted(sub[gender_col].unique())]
    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=False)
    title = f"{col} by {gender_col}" + (f" ({tag})" if tag else "")
    plt.title(title)
    plt.xlabel(gender_col)
    plt.ylabel(col)
    plt.tight_layout()
    fname = os.path.join(outdir, f"box_{col}_by_{gender_col}" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_trend_by_bloc(df, col, outdir, bloc_col="bloc", tag=None):
    if bloc_col not in df.columns or col not in df.columns:
        return None
    sub = df[[bloc_col, col]].dropna()
    if sub.empty:
        return None
    agg = sub.groupby(bloc_col)[col].agg(['mean', 'std', 'count']).reset_index()
    x = agg[bloc_col].values
    y = agg['mean'].values
    std = agg['std'].values
    plt.figure()
    plt.plot(x, y)
    lo = y - std
    hi = y + std
    try:
        plt.fill_between(x, lo, hi, alpha=0.2)
    except Exception:
        pass
    title = f"{col} Trend by {bloc_col}" + (f" ({tag})" if tag else "")
    plt.title(title)
    plt.xlabel(bloc_col)
    plt.ylabel(col)
    plt.tight_layout()
    fname = os.path.join(outdir, f"trend_{col}_by_{bloc_col}" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ---------------------------- correlations & stats ----------------------------
def correlation_heatmap(df, cols, outdir, method="pearson", sample_n=80000, random_state=42, tag="subset"):
    data = df[cols].dropna()
    if data.empty:
        return None, None
    if sample_n and len(data) > sample_n:
        data = data.sample(sample_n, random_state=random_state)
    corr = data.corr(method=method)

    plt.figure()
    plt.imshow(corr.values, interpolation='nearest')
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title(f"{method.capitalize()} Correlation Heatmap ({tag})")
    plt.colorbar()
    plt.tight_layout()
    fname = os.path.join(outdir, f"corr_heatmap_{tag}_{method}.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    return corr, fname


def top_corr_with(df, target, cols, method="pearson", k=12, sample_n=80000, random_state=42):
    valid_cols = [c for c in cols if c in df.columns]
    if target not in df.columns or not valid_cols:
        return pd.DataFrame(columns=["var", "r", "p", "n"])
    data = df[valid_cols + [target]].dropna()
    if data.empty:
        return pd.DataFrame(columns=["var", "r", "p", "n"])
    if sample_n and len(data) > sample_n:
        data = data.sample(sample_n, random_state=random_state)
    res = []
    for c in valid_cols:
        if c == target:
            continue
        try:
            if method == "pearson":
                r, p = stats.pearsonr(data[c], data[target])
            else:
                r, p = stats.spearmanr(data[c], data[target])
            n = len(data[[c, target]].dropna())
            res.append((c, r, p, n))
        except Exception:
            continue
    out = pd.DataFrame(res, columns=["var", "r", "p", "n"]).sort_values("r", key=lambda s: s.abs(), ascending=False)
    return out.head(k)


def point_biserial_against_gender(df, cols, gender_col="gender", sample_n=80000, random_state=42):
    if gender_col not in df.columns:
        return pd.DataFrame(columns=["var", "r_pb", "p", "n"])
    valid_cols = [c for c in cols if c in df.columns]
    sub = df[[gender_col] + valid_cols].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["var", "r_pb", "p", "n"])
    if sample_n and len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=random_state)
    res = []
    for c in valid_cols:
        try:
            r, p = stats.pointbiserialr(sub[gender_col], sub[c])
            n = len(sub[[gender_col, c]].dropna())
            res.append((c, r, p, n))
        except Exception:
            continue
    out = pd.DataFrame(res, columns=["var", "r_pb", "p", "n"]).sort_values("r_pb", key=lambda s: s.abs(),
                                                                           ascending=False)
    return out


def compute_vif(df, cols, sample_n=20000, random_state=42):
    valid_cols = [c for c in cols if c in df.columns]
    X = df[valid_cols].dropna()
    if X.empty:
        return pd.DataFrame(columns=["Variable", "VIF"])
    if sample_n and len(X) > sample_n:
        X = X.sample(sample_n, random_state=random_state)
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


# ---------------------------- special plots ----------------------------
def plot_age_mortality_dedup(df, outdir, dedup_key="icustayid"):
    if dedup_key not in df.columns or "age" not in df.columns or "morta_90" not in df.columns:
        print(f"[WARN] columns missing for age-mortality plot.")
        return None
    df_demo = df.drop_duplicates(subset=dedup_key).copy()

    bins = [0,20,30,40,50,60,70,80,90,120]
    labels = ["0-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90+"]
    df_demo["age_bin"] = pd.cut(df_demo["age"], bins=bins, labels=labels, right=False)

    mortality_by_age = df_demo.groupby("age_bin")["morta_90"].mean()
    count_by_age = df_demo["age_bin"].value_counts().sort_index()

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(mortality_by_age.index.astype(str), mortality_by_age.values, marker="o")
    ax1.set_ylabel("90-day mortality rate")
    ax1.set_xlabel("Age group")
    ax1.set_ylim(0, 1)
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(count_by_age.index.astype(str), count_by_age.values, alpha=0.3)
    ax2.set_ylabel("Patient count")

    plt.title(f"Age vs 90-day Mortality (dedup by {dedup_key})")
    fig.tight_layout()

    ensure_dir(outdir)
    outpath = os.path.join(outdir, f"age_mortality_dedup_{dedup_key}.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def plot_sofa_vs_mortality_boxplot(df, outdir, dedup_key=None, tag=None):
    if "SOFA" not in df.columns or "morta_90" not in df.columns:
        print("[WARN] Columns 'SOFA' or 'morta_90' not found. Skip SOFA vs mortality boxplot.")
        return None

    df_use = df
    tag_local = tag
    if dedup_key is not None and dedup_key in df.columns:
        df_use = df.drop_duplicates(subset=dedup_key).copy()
        tag_local = (tag + "_" if tag else "") + f"dedup_{dedup_key}"

    sub = df_use[["SOFA", "morta_90"]].dropna()
    if sub.empty or sub["morta_90"].nunique() < 2:
        print("[WARN] Not enough data to plot SOFA vs mortality boxplot.")
        return None

    g0 = sub[sub["morta_90"] == 0]["SOFA"].values
    g1 = sub[sub["morta_90"] == 1]["SOFA"].values

    plt.figure()
    plt.boxplot([g0, g1], labels=["Survivors", "Non-survivors"], showfliers=False)
    title = "SOFA vs 90-day Mortality" + (f" ({tag_local})" if tag_local else "")
    plt.title(title)
    plt.ylabel("SOFA")
    plt.tight_layout()

    ensure_dir(outdir)
    fname = os.path.join(outdir, "sofa_mortality_boxplot" + (f"_{tag_local}" if tag_local else "") + ".png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_sofa_binned_mortality_curve(df, outdir, dedup_key=None, tag=None):
    if "SOFA" not in df.columns or "morta_90" not in df.columns:
        print("[WARN] Columns 'SOFA' or 'morta_90' not found. Skip binned curve.")
        return None

    df_use = df
    tag_local = tag
    if dedup_key is not None and dedup_key in df.columns:
        df_use = df.drop_duplicates(subset=dedup_key).copy()
        tag_local = (tag + "_" if tag else "") + f"dedup_{dedup_key}"

    sub = df_use[["SOFA", "morta_90"]].dropna()
    if sub.empty:
        print("[WARN] Not enough data to plot binned mortality curve.")
        return None

    bins = [-0.5, 3, 6, 9, 12, 20]
    labels = ["0–3", "4–6", "7–9", "10–12", "13+"]
    sub["SOFA_bin"] = pd.cut(sub["SOFA"], bins=bins, labels=labels)

    mort = sub.groupby("SOFA_bin")["morta_90"].mean()
    cnt = sub["SOFA_bin"].value_counts().sort_index()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(mort.index.astype(str), mort.values, marker="o")
    ax1.set_ylabel("90-day mortality rate")
    ax1.set_xlabel("SOFA")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.bar(cnt.index.astype(str), cnt.values, alpha=0.3)
    ax2.set_ylabel("Patient count")

    title = "Mortality by SOFA bin" + (f" ({tag_local})" if tag_local else "")
    plt.title(title)
    plt.xticks(rotation=0)
    fig.tight_layout()

    ensure_dir(outdir)
    fname = os.path.join(outdir, "sofa_mortality_binned" + (f"_{tag_local}" if tag_local else "") + ".png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ---------------------------- patient-level aggregation ----------------------------
def first_non_null(series):
    for v in series:
        if pd.notna(v):
            return v
    return np.nan


def make_patient_level_table(df, id_col="icustayid", numeric_strategy="mean"):
    """
    Aggregate multiple rows per id_col into one row:
    - continuous numeric: mean
    - binary numeric ({0,1}): max
    - non-numeric: first non-null
    NOTE: id_col is excluded from aggregation and kept as a key.
    """
    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' not in columns.")

    df2 = df.copy()

    # Build column sets while EXCLUDING id_col
    all_numeric = df2.select_dtypes(include=[np.number]).columns.tolist()
    if id_col in all_numeric:
        all_numeric.remove(id_col)
    cat_cols = [c for c in df2.columns if c not in all_numeric and c != id_col]

    # Identify binary-like numeric columns
    bin_like = []
    for c in all_numeric:
        vals = pd.unique(df2[c].dropna())
        if len(vals) > 0 and set(vals).issubset({0, 1}):
            bin_like.append(c)
    cont_nums = [c for c in all_numeric if c not in bin_like]

    # Aggregations
    agg_num = {c: "mean" for c in cont_nums}  # continuous: mean
    for c in bin_like:
        agg_num[c] = "max"  # binary: max (any 1 → 1)
    agg_cat = {c: first_non_null for c in cat_cols}

    agg_dict = {**agg_num, **agg_cat}

    # Group and aggregate (id_col is only the key, NOT aggregated)
    patient_df = df2.groupby(id_col, as_index=False).agg(agg_dict)
    return patient_df


def run_patient_level_univariates(df_patient, outdir, tag):
    # keep a compact list to avoid too many files; adjust as needed
    core_numeric = [
        "HR","SysBP","DiasBP","MeanBP","RR","SpO2","Temp","Glucose",
        "Creatinine","Bilirubin","Platelets","WBC","PaO2_FiO2",
        "SOFA","Shock_Index","cumulated_balance","age"
    ]
    core_numeric = [c for c in core_numeric if c in df_patient.columns]
    for col in core_numeric:
        plot_hist(df_patient, col, outdir, tag=f"{tag}_patientmean")

    # gender boxplots at patient level
    if "gender" in df_patient.columns:
        for col in [c for c in core_numeric if c != "gender"]:
            plot_box_by_gender(df_patient, col, outdir, gender_col="gender", tag=f"{tag}_patientmean")


def run_patient_level_correlations(df_patient, outdir, tag):
    corr_subset = [c for c in [
        "HR","MeanBP","RR","SpO2","Temp","Glucose","Creatinine","Bilirubin",
        "Platelets","WBC","PaO2_FiO2","SOFA","Shock_Index","cumulated_balance","age"
    ] if c in df_patient.columns]
    if len(corr_subset) >= 2:
        correlation_heatmap(df_patient, corr_subset, outdir, method="pearson", tag=f"{tag}_patientmean")
        correlation_heatmap(df_patient, corr_subset, outdir, method="spearman", tag=f"{tag}_patientmean")

    core_vars = [c for c in [
        "HR","SysBP","DiasBP","MeanBP","RR","SpO2","Temp",
        "Glucose","Creatinine","Bilirubin","Platelets",
        "SOFA","Shock_Index"
    ] if c in df_patient.columns]
    if core_vars:
        vif_df = compute_vif(df_patient, core_vars, sample_n=20000)
        out_csv = os.path.join(outdir, f"vif_core_subset_{tag}_patientmean.csv")
        vif_df.to_csv(out_csv, index=False)


def run_patient_level_outcome_plots(df_patient, outdir, tag):
    if "SOFA" in df_patient.columns and "morta_90" in df_patient.columns:
        plot_sofa_vs_mortality_boxplot(df_patient, outdir, dedup_key=None, tag=f"{tag}_patientmean")
        plot_sofa_binned_mortality_curve(df_patient, outdir, dedup_key=None, tag=f"{tag}_patientmean")


# ---------------------------- main ----------------------------
def main(args):
    df = pd.read_csv(args.csv)
    ensure_dir(args.outdir)

    # Optional patient-level dedup for demographics
    df_demo = df
    tag = None
    if args.dedup and args.dedup in df.columns:
        df_demo = df.drop_duplicates(subset=args.dedup)
        tag = f"dedup_{args.dedup}"
    elif args.dedup:
        print(f"[WARN] dedup column '{args.dedup}' not found. Demographic plots will use full rows.")

    # Column candidates
    vitals_candidates = ["HR", "SysBP", "DiasBP", "MeanBP", "RR", "Temp", "SpO2"]
    labs_candidates = ["Lactate", "Glucose", "pH", "PaCO2", "PaO2", "FiO2", "PaO2_FiO2",
                       "BUN", "Creatinine", "Platelets", "WBC", "Hematocrit", "Hemoglobin",
                       "Bilirubin", "Sodium", "Potassium", "Chloride", "Magnesium", "Calcium"]
    resp_candidates = ["PEEP", "PlateauPres"]
    scores_candidates = ["SOFA", "SIRS", "Shock_Index"]
    fluids_candidates = ["Urine", "cumulated_balance"]

    vitals = safe_cols(df, vitals_candidates)
    labs = safe_cols(df, labs_candidates)
    resp = safe_cols(df, resp_candidates)
    scores = safe_cols(df, scores_candidates)
    fluids = safe_cols(df, fluids_candidates)

    # 1) Demographics on df_demo (dedup if requested)
    if "age" in df_demo.columns:
        plot_hist(df_demo, "age", args.outdir, tag=tag)
    if "gender" in df_demo.columns:
        plot_bar_counts(df_demo, "gender", args.outdir, tag=tag)

    # 1b) Univariate & gender boxplots on full df (measurement-level)
    subset_for_plots = vitals + labs[:4] + scores + fluids
    for col in subset_for_plots:
        plot_hist(df, col, args.outdir)
        if "gender" in df.columns:
            plot_box_by_gender(df, col, args.outdir, gender_col="gender")

    # 2) Trend by bloc (full df)
    trend_cols = safe_cols(df, ["HR", "MeanBP", "RR", "SpO2", "Lactate", "PaO2_FiO2", "Shock_Index", "SOFA",
                                "cumulated_balance"])
    for col in trend_cols:
        plot_trend_by_bloc(df, col, args.outdir, bloc_col="bloc")

    # 3) Correlation heatmaps (full df)
    corr_subset = list(
        dict.fromkeys(safe_cols(df, ["HR", "MeanBP", "RR", "SpO2", "Temp", "Lactate", "Glucose", "PaCO2", "PaO2",
                                     "FiO2", "PaO2_FiO2", "SOFA", "SIRS", "Shock_Index", "Urine", "cumulated_balance",
                                     "age"])))
    if len(corr_subset) >= 2:
        correlation_heatmap(df, corr_subset, args.outdir, method="pearson", tag="clinical_core")
        correlation_heatmap(df, corr_subset, args.outdir, method="spearman", tag="clinical_core")

    # 4) Top-K vs key scores (full df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop_col in ["icustayid", "timestep"]:
        if drop_col in numeric_cols:
            numeric_cols.remove(drop_col)
    for target in safe_cols(df, ["SOFA", "SIRS", "Shock_Index"]):
        top_corr_with(df, target, [c for c in numeric_cols if c != target], method="pearson", k=15).to_csv(
            os.path.join(args.outdir, f"topcorr_{target}_pearson.csv"), index=False
        )
        top_corr_with(df, target, [c for c in numeric_cols if c != target], method="spearman", k=15).to_csv(
            os.path.join(args.outdir, f"topcorr_{target}_spearman.csv"), index=False
        )

    # 5) Point-biserial vs gender (full df)
    if "gender" in df.columns:
        pb = point_biserial_against_gender(df, [c for c in numeric_cols if c != "gender"], gender_col="gender")
        pb.to_csv(os.path.join(args.outdir, "pointbiserial_gender_vs_numeric.csv"), index=False)

    # 6) VIF on a core subset (full df; sampling inside)
    core_vars = safe_cols(df, ["HR", "SysBP", "DiasBP", "MeanBP", "RR", "SpO2", "Temp",
                               "Lactate", "Glucose", "Creatinine", "Platelets", "Bilirubin",
                               "SOFA", "SIRS", "Shock_Index"])
    if core_vars:
        vif_df = compute_vif(df, core_vars, sample_n=20000)
        vif_df.to_csv(os.path.join(args.outdir, "vif_core_subset.csv"), index=False)

    # 7) Age vs Mortality (dedup-level view already)
    if args.dedup and args.dedup in df.columns and "morta_90" in df.columns:
        plot_age_mortality_dedup(df, args.outdir, dedup_key=args.dedup)

    # 8) SOFA vs mortality (raw + optional dedup overlay)
    if "SOFA" in df.columns and "morta_90" in df.columns:
        plot_sofa_vs_mortality_boxplot(df, args.outdir, dedup_key=args.dedup if args.dedup in df.columns else None)
        plot_sofa_binned_mortality_curve(df, args.outdir, dedup_key=args.dedup if args.dedup in df.columns else None)

    # ===== Patient-level (mean) views — enabled by default =====
    if args.dedup and args.dedup in df.columns:
        try:
            df_patient = make_patient_level_table(df, id_col=args.dedup, numeric_strategy="mean")
            tag_pm = f"{args.dedup}"

            # patient-level univariates & gender boxplots
            run_patient_level_univariates(df_patient, args.outdir, tag_pm)

            # patient-level correlations & VIF
            run_patient_level_correlations(df_patient, args.outdir, tag_pm)

            # patient-level SOFA vs mortality
            run_patient_level_outcome_plots(df_patient, args.outdir, tag_pm)

        except Exception as e:
            print(f"[WARN] patient-level aggregation failed: {e}")

    # index of outputs
    with open(os.path.join(args.outdir, "index.txt"), "w", encoding="utf-8") as f:
        for name in sorted(os.listdir(args.outdir)):
            if name.endswith(".png") or name.endswith(".csv"):
                f.write(name + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="mimic_dataset.csv", help="Path to mimic_dataset.csv")
    parser.add_argument("--outdir", type=str, default="./eda_outputs", help="Output directory for figures & tables")
    parser.add_argument("--dedup", type=str, default="icustayid",
                        help="Column name used as patient/stay id for patient-level views (default: icustayid)")
    args = parser.parse_args()
    main(args)
