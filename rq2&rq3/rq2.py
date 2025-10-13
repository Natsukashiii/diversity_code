# -*- coding: utf-8 -*-
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from services.rq2_helper import build_monthly_metrics_for_repo
from tqdm import tqdm

from config.load_config import load_config

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================
# ------------------ CONFIG ------------------
# =====================================================
cfg = load_config()
RQ1_DATA_PATH = Path(str(cfg.path.rq1_path))
RQ2_DATA_PATH = Path(str(cfg.path.rq2_path))
DATASET_INFO_DIR = Path(str(cfg.path.dataset_info_path))

BASE_CSV = RQ1_DATA_PATH / "base.csv"
QUOTA_CSV = RQ1_DATA_PATH / "quota.csv"
OUT_CSV = RQ2_DATA_PATH / "rq2_distribution.csv"       
VALUES_CSV = RQ2_DATA_PATH / "rq2_metric_values.csv"  

# =====================================================
# ------------------ METRICS ------------------
# =====================================================
METRICS = [
    "commits_per_month",
    "files_changed_per_commit",
    "avg_churn_per_commit",
    "issues_opened",
    "issues_closed",
    "prs_opened",
    "prs_closed",
    "pr_latency_hours_avg",
    "pr_merge_ratio",
    "merge_ratio",
    "releases_per_month",
    "bug_issues",
    "bug_issues_core_dev",
    "bug_issues_external",
]

# =====================================================
# ------------------ Helper Functions -----------------
# =====================================================

def cliffs_delta(a, b):
    """Compute Cliff's Delta effect size."""
    a = pd.Series(a).dropna().to_numpy()
    b = pd.Series(b).dropna().to_numpy()
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    greater = sum((x > b).sum() for x in a)
    less = sum((x < b).sum() for x in a)
    return (greater - less) / (n * m)


def summarize_repo_set(repo_list_csv: Path, dataset_label: str) -> pd.DataFrame:
    """Aggregate monthly metrics for all repos in one dataset."""
    df = pd.read_csv(repo_list_csv)
    repo_names = df["repo"].unique()
    all_data = []

    for repo in tqdm(repo_names, desc=f"Processing {dataset_label}"):
        repo_dir = DATASET_INFO_DIR / repo
        if not repo_dir.exists():
            print(f"[WARN] Missing data dir for {repo}")
            continue

        monthly_df = build_monthly_metrics_for_repo(repo, repo_dir)
        if monthly_df.empty:
            continue
        monthly_df["dataset"] = dataset_label
        monthly_df["repo"] = repo
        all_data.append(monthly_df)

    if not all_data:
        print(f"[WARN] No valid data for {dataset_label}")
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def compute_descriptive_stats(series: pd.Series) -> dict:
    """Compute key descriptive statistics."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {k: np.nan for k in ["mean", "median", "std", "min", "max", "p25", "p75"]}
    return {
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
        "p25": s.quantile(0.25),
        "p75": s.quantile(0.75),
    }


def compare_distributions(base_df: pd.DataFrame, quota_df: pd.DataFrame, metric: str) -> dict:
    """Compare metric distributions between Stratified and Quota datasets."""
    if metric not in base_df.columns or metric not in quota_df.columns:
        print(f"[WARN] {metric} missing in one dataset, skipping.")
        return None

    a = pd.to_numeric(base_df[metric], errors="coerce").dropna()
    b = pd.to_numeric(quota_df[metric], errors="coerce").dropna()
    if a.empty or b.empty:
        return None

    stats_a = compute_descriptive_stats(a)
    stats_b = compute_descriptive_stats(b)

    # Mann–Whitney U test
    try:
        u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        u_stat, p_val = np.nan, np.nan

    # Cliff’s delta
    delta = cliffs_delta(a, b)

    # Relative differences
    mean_diff_pct = (stats_b["mean"] - stats_a["mean"]) / stats_a["mean"] * 100 if stats_a["mean"] else np.nan
    median_diff_pct = (stats_b["median"] - stats_a["median"]) / stats_a["median"] * 100 if stats_a["median"] else np.nan

    result = {
        "metric": metric,
        "base_mean": stats_a["mean"],
        "quota_mean": stats_b["mean"],
        "mean_diff_pct": mean_diff_pct,
        "base_median": stats_a["median"],
        "quota_median": stats_b["median"],
        "median_diff_pct": median_diff_pct,
        "base_std": stats_a["std"],
        "quota_std": stats_b["std"],
        "base_min": stats_a["min"],
        "quota_min": stats_b["min"],
        "base_max": stats_a["max"],
        "quota_max": stats_b["max"],
        "base_p25": stats_a["p25"],
        "quota_p25": stats_b["p25"],
        "base_p75": stats_a["p75"],
        "quota_p75": stats_b["p75"],
        "mannwhitney_U": u_stat,
        "p_value": p_val,
        "cliffs_delta": delta,
    }
    return result


def melt_metric_values(df: pd.DataFrame, dataset_label: str, metrics: list) -> pd.DataFrame:
    keep_cols = ["repo", "month"] + [m for m in metrics if m in df.columns]
    sub = df[keep_cols].copy()
    long_df = sub.melt(id_vars=["repo", "month"], var_name="metric", value_name="value")
    long_df["dataset"] = dataset_label
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    return long_df[["dataset", "metric", "value", "repo", "month"]]

# ================== Plotting (read from VALUES_CSV) ==================
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _winsorize_array(arr: np.ndarray, q=(0.01, 0.99)) -> np.ndarray:
    if arr.size == 0 or q is None:
        return arr
    lo, hi = np.nanquantile(arr, q[0]), np.nanquantile(arr, q[1])
    return np.clip(arr, lo, hi)

def _prepare_hist_bins(data_arrays, use_log_x: bool, is_count_like: bool, bins: int = 50):
    all_x = np.concatenate([x for x in data_arrays if x.size])
    if all_x.size == 0:
        return bins
    if use_log_x:
        all_x = all_x[all_x > 0]
        if all_x.size == 0:
            return bins
        xmin = float(np.nanmin(all_x)); xmax = float(np.nanmax(all_x))
        if xmax <= xmin: xmax = xmin * 1.1
        return np.logspace(np.log10(xmin), np.log10(xmax), bins)
    else:
        if is_count_like:
            vmin = float(np.nanmin(all_x)); vmax = float(np.nanmax(all_x))
            return np.arange(np.floor(vmin) - 0.5, np.ceil(vmax) + 0.5 + 1e-9, 1.0)
        else:
            xmin = float(np.nanmin(all_x)); xmax = float(np.nanmax(all_x))
            if xmax <= xmin: xmax = xmin + 1.0
            return np.linspace(xmin, xmax, bins)

def plot_distribution_overlap_arrays(
    base_vals: np.ndarray,
    quota_vals: np.ndarray,
    title: str,
    *,
    use_log_x: bool = True,
    winsorize_q = (0.01, 0.99),
    bins: int = 50,
    density: bool = True,
    figsize=(6.0, 4.0),
    alpha_base=0.45,
    alpha_quota=0.35,
):
    base = pd.to_numeric(pd.Series(base_vals), errors="coerce").dropna().to_numpy()
    quota = pd.to_numeric(pd.Series(quota_vals), errors="coerce").dropna().to_numpy()
    base = _winsorize_array(base, winsorize_q)
    quota = _winsorize_array(quota, winsorize_q)

    is_count_like = False
    if base.size + quota.size:
        sample = np.concatenate([base[:2000], quota[:2000]])
        if sample.size:
            frac_int = np.mean(np.isclose(sample, np.round(sample)))
            is_count_like = bool(frac_int > 0.9)

    cur_bins = _prepare_hist_bins([base, quota], use_log_x, is_count_like, bins=bins)

    fig, ax = plt.subplots(figsize=figsize)
    if use_log_x:
        ax.set_xscale("log")

    ax.hist(base,  bins=cur_bins, density=density, alpha=alpha_base,
            label="Stratified", color="#6baed6", edgecolor="none")
    ax.hist(quota, bins=cur_bins, density=density, alpha=alpha_quota,
            label="Quota",       color="#fd8d3c", edgecolor="none")

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig

def plot_all_metrics_from_values_csv(
    values_csv: Path,
    metrics: list,
    out_pdf: Path,
    *,
    use_log_x: bool = True,
    winsorize_q=(0.01, 0.99),
    bins: int = 50,
    density: bool = True,
    figsize=(6.2, 4.2),
):
    if not Path(values_csv).exists():
        raise FileNotFoundError(f"Values CSV not found: {values_csv}")

    vals = pd.read_csv(values_csv, low_memory=False)
    required = {"dataset", "metric", "value"}
    missing = required - set(vals.columns)
    if missing:
        raise ValueError(f"{values_csv} is missing columns: {missing}")

    vals["dataset"] = vals["dataset"].astype(str)
    vals["metric"] = vals["metric"].astype(str)
    vals["value"] = pd.to_numeric(vals["value"], errors="coerce")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for m in metrics:
            sub = vals[vals["metric"] == m]
            if sub.empty:
                continue
            base = sub[sub["dataset"].str.lower().str.contains("stratified")]["value"].dropna().to_numpy()
            quota = sub[sub["dataset"].str.lower().str.contains("quota")]["value"].dropna().to_numpy()
            if base.size == 0 and quota.size == 0:
                continue

            fig = plot_distribution_overlap_arrays(
                base, quota, m.replace("_", r"\_"),
                use_log_x=use_log_x,
                winsorize_q=winsorize_q,
                bins=bins,
                density=density,
                figsize=figsize
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

# =====================================================
# ------------------ Main Entry -----------------
# =====================================================

def main():
    # --- if VALUES_CSV exist ---
    if VALUES_CSV.exists() and OUT_CSV.exists():
        print(f"[INFO] Found existing files:\n  - {OUT_CSV}\n  - {VALUES_CSV}\n[INFO] Will only render plots from values CSV.")
        dist_pdf = RQ2_DATA_PATH / "rq2_distributions.pdf"
        plot_all_metrics_from_values_csv(
            VALUES_CSV, METRICS, dist_pdf,
            use_log_x=True,
            winsorize_q=(0.01, 0.99),
            bins=50,
            density=True,
            figsize=(6.2, 4.2),
        )
        print(f"[OK] Distribution plots saved to {dist_pdf}")
        return

    print("[INFO] Loading datasets & building monthly metrics...")
    base_df = summarize_repo_set(BASE_CSV, "Stratified")
    quota_df = summarize_repo_set(QUOTA_CSV, "Quota")

    if base_df.empty or quota_df.empty:
        print("[ERROR] Empty dataset(s). Check data directories.")
        return

    # -------- write OUT_CSV --------
    print("[INFO] Comparing distributions...")
    results = []
    for metric in tqdm(METRICS, desc="Comparing Metrics"):
        res = compare_distributions(base_df, quota_df, metric)
        if res:
            results.append(res)

    result_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved stats summary to {OUT_CSV}")

    # -------- save --------
    print("[INFO] Saving per-observation values (long table) for plotting...")
    base_long  = melt_metric_values(base_df,  "Stratified", METRICS)
    quota_long = melt_metric_values(quota_df, "Quota",      METRICS)
    values_long = pd.concat([base_long, quota_long], ignore_index=True)
    values_long.to_csv(VALUES_CSV, index=False)
    print(f"[OK] Saved metric values to {VALUES_CSV}")

    # -------- plot --------
    dist_pdf = RQ2_DATA_PATH / "rq2_distributions.pdf"
    plot_all_metrics_from_values_csv(
        VALUES_CSV, METRICS, dist_pdf,
        use_log_x=True, winsorize_q=(0.01, 0.99), bins=50, density=True, figsize=(6.2, 4.2),
    )
    print(f"[OK] Distribution plots saved to {dist_pdf}")


if __name__ == "__main__":
    main()