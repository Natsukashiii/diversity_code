# rq2_helper.py
# Build time-bucketed (month/week) repo metrics, split by pre/post CI, summarize, and plot.

import json
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon

__all__ = [
    "build_monthly_metrics_for_repo",
    "label_pre_post",
    "compute_cohort_distributions",
    "summarize_pre_post",
    "plot_distributions_pdf",
]

# ------------------ Granularity switch ------------------
# "month" -> monthly buckets; "week" -> weekly buckets (week starts on Monday).
GRANULARITY: str = "month"   # change to "week" month to switch globally

def _period_code() -> str:
    """Return pandas period code for current GRANULARITY."""
    return "M" if GRANULARITY.lower() == "month" else "W-MON"

def _period_label() -> str:
    """Human-friendly label for charts and logs."""
    return "Month" if GRANULARITY.lower() == "month" else "Week"

# ------------------ Utilities ------------------

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise safe division → NaN if denom<=0 or NaN."""
    a = pd.to_numeric(num, errors="coerce")
    b = pd.to_numeric(den, errors="coerce")
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV if present; else return empty DataFrame."""
    if path is None or not path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()

def _to_bucket(dt_series: pd.Series) -> pd.Series:
    """
    Normalize timestamps to bucket start (month-start or week-start Monday),
    return naive datetime64[ns]. Keeps original length/index.
    """
    s = pd.Series(dt_series).copy()
    s = pd.to_datetime(s, errors="coerce", utc=True)

    if s.isna().all():
        return pd.Series([pd.NaT] * len(s), index=s.index)

    # tz-aware -> naive
    idx = pd.DatetimeIndex(s).tz_convert("UTC").tz_localize(None)
    # to period and back to timestamp at period start
    pcode = _period_code()
    bucket_idx = idx.to_period(pcode).to_timestamp()
    return pd.Series(bucket_idx.values, index=s.index)

def _normalize_bucket_column(df: pd.DataFrame, col: str = "month") -> pd.DataFrame:
    """
    Force df[col] to period-start timestamps (naive), honoring current GRANULARITY.
    Column name kept as 'month' to minimize code changes.
    """
    if df is None or df.empty or col not in df.columns:
        return df
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        df[col] = pd.to_datetime(s, errors="coerce")
        df[col] = df[col].dt.to_period(_period_code()).dt.to_timestamp()
        return df

    # period dtype
    if isinstance(getattr(s, "dtype", None), pd.PeriodDtype):
        df[col] = s.dt.to_timestamp()
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.to_period(_period_code()).dt.to_timestamp()
        return df

    # integer like 202101 -> parse as YYYYMM (only meaningful for month); fallback to epoch
    parsed = pd.to_datetime(s, errors="coerce")
    df[col] = parsed.dt.to_period(_period_code()).dt.to_timestamp()
    return df

# ------------------ Metric builders (per repo) ------------------

def _commits_metrics(commits_csv: Path) -> pd.DataFrame:
    """
    Return bucketed commit metrics:
      - commits_per_month
      - avg_churn_per_commit
      - files_changed_per_commit   
      - files_changed_per_month       
      - merge_ratio                    <-- merge commits / all commits
    Expected columns: author_date, insertions, deletions, (optional) files_changed, is_merge/parents_count
    """
    df = _safe_read_csv(commits_csv)
    if df.empty or "author_date" not in df.columns:
        return pd.DataFrame(columns=[
            "month", "commits_per_month", "avg_churn_per_commit",
            "files_changed_per_commit", "files_changed_per_month", "merge_ratio"
        ])

    dates = pd.to_datetime(df["author_date"], errors="coerce", utc=True)
    if dates.isna().all():
        return pd.DataFrame(columns=[
            "month", "commits_per_month", "avg_churn_per_commit",
            "files_changed_per_commit", "files_changed_per_month", "merge_ratio"
        ])

    df["month"] = _to_bucket(dates)

    df["insertions"] = pd.to_numeric(df.get("insertions", 0), errors="coerce").fillna(0)
    df["deletions"]  = pd.to_numeric(df.get("deletions", 0),  errors="coerce").fillna(0)
    df["churn"]      = df["insertions"] + df["deletions"]

    df["files_changed"] = pd.to_numeric(df.get("files_changed", 0), errors="coerce").fillna(0)

    if "is_merge" in df.columns:
        df["is_merge"] = df["is_merge"].astype(bool)
    else:
        pc = pd.to_numeric(df.get("parents_count", np.nan), errors="coerce")
        df["is_merge"] = pc.fillna(1) >= 2

    grp = df.groupby("month", as_index=False).agg(
        commits_per_month=("author_date", "count"),
        avg_churn_per_commit=("churn", "mean"),
        files_changed_per_commit=("files_changed", "mean"),  # ✅ 每次提交平均修改文件数（按月平均）
        files_changed_per_month=("files_changed", "sum"),        # 可选：本月总修改文件数
        merge_commits=("is_merge", "sum"),
    )
    grp["merge_ratio"] = grp["merge_commits"] / grp["commits_per_month"]
    return grp

def _parse_labels_has_bug(cell) -> bool:
    """Return True if labels contain 'bug' (case-insensitive)."""
    if pd.isna(cell):
        return False
    try:
        obj = json.loads(cell)
        if isinstance(obj, list):
            vals = " ".join(str(x.get("name", "")) for x in obj)
        else:
            vals = str(obj)
    except Exception:
        vals = str(cell)
    return "bug" in vals.lower()

def _is_core_dev(assoc) -> bool:
    """Core developer if association in GitHub internal roles."""
    core_roles = {"MEMBER", "OWNER", "COLLABORATOR"}
    return isinstance(assoc, str) and assoc.upper() in core_roles

def _issues_metrics(issues_csv: Path) -> pd.DataFrame:
    """
    Return bucketed opened/closed issues and bug splits.
    Bug = labels contain 'bug'.
    Core dev = author_association in {MEMBER, OWNER, COLLABORATOR}.
    """
    df = _safe_read_csv(issues_csv)
    if df.empty:
        cols = ["month", "issues_opened", "issues_closed", "bug_issues", "bug_issues_core_dev", "bug_issues_external"]
        return pd.DataFrame(columns=cols)

    # ——if author_association missing, skip bug/core splits——
    if "author_association" not in df.columns:
        logging.warning(
            "[rq2_helper] 'author_association' missing in issues.csv → skip repo for issue-derived metrics."
        )
        return pd.DataFrame()  

    df["created_month"] = _to_bucket(df.get("created_at"))
    df["closed_month"]  = _to_bucket(df.get("closed_at"))
    df["is_bug"] = df["labels"].apply(_parse_labels_has_bug)
    df["is_core_dev"] = df["author_association"].apply(_is_core_dev)

    opened = (
        df.groupby("created_month", as_index=False)
          .agg(
              issues_opened=("id", "count"),
                bug_issues=("is_bug", "sum"),
              bug_issues_core_dev=("is_bug", lambda s: int(((s) & df.loc[s.index, "is_core_dev"]).sum())),
              bug_issues_external=("is_bug", lambda s: int(((s) & (~df.loc[s.index, "is_core_dev"])).sum())),
          )
          .rename(columns={"created_month": "month"})
    )

    closed = (
        df[~df["closed_month"].isna()]
        .groupby("closed_month", as_index=False)
        .agg(issues_closed=("id", "count"))
        .rename(columns={"closed_month": "month"})
    )

    out = pd.merge(opened, closed, on="month", how="outer").sort_values("month")
    for col in ["issues_opened", "issues_closed", "bug_issues", "bug_issues_core_dev", "bug_issues_external"]:
        if col in out.columns:
            # keep as float for consistency; no rounding
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["issue_close_rate"] = _safe_div(out["issues_closed"], out["issues_opened"])
    return out

def _prs_metrics(prs_csv: Path) -> pd.DataFrame:
    """
    Return bucketed PRs opened/closed, avg close latency (hours), and merge ratio.
    - Opened grouped by created_at bucket.
    - Latency, merge ratio grouped by closed_at bucket.
    """
    df = _safe_read_csv(prs_csv)
    if df.empty:
        cols = ["month", "prs_opened", "prs_closed", "pr_latency_hours_avg", "pr_merge_ratio","merge_ratio"]
        return pd.DataFrame(columns=cols)

    if "id" not in df.columns:
        df = df.copy()
        df["id"] = np.arange(len(df), dtype=np.int64)
        logging.warning("[rq2_helper] No 'id' column in %s, using synthetic ids.", getattr(prs_csv, "name", "prs.csv"))


    df["created_at"] = pd.to_datetime(
    df.get("created_at"),
    errors="coerce",
    utc=True,
    format="%Y-%m-%dT%H:%M:%SZ"  
)
    df["closed_at"]  = pd.to_datetime(df.get("closed_at"),  errors="coerce", utc=True)
    df["merged_at"]  = pd.to_datetime(df.get("merged_at"),  errors="coerce", utc=True)

    created_bucket = _to_bucket(df["created_at"])
    closed_bucket  = _to_bucket(df["closed_at"])

    opened = (
        pd.DataFrame({"month": created_bucket})
        .assign(id=df["id"])
        .dropna(subset=["month"])
        .groupby("month", as_index=False)
        .agg(prs_opened=("id", "count"))
    )

    closed_df = df[~df["closed_at"].isna()].copy()
    if closed_df.empty:
        closed = pd.DataFrame(columns=["month", "prs_closed", "pr_latency_hours_avg", "pr_merge_ratio","merge_ratio"])
    else:
        closed_df["month"] = closed_bucket.loc[closed_df.index].values
        closed_df = closed_df.dropna(subset=["month"]).copy()
        closed_df["latency_hours"] = (closed_df["closed_at"] - closed_df["created_at"]).dt.total_seconds() / 3600.0
        closed_df["is_merged"] = ~closed_df["merged_at"].isna()

        closed = (
            closed_df.groupby("month", as_index=False)
            .agg(
                prs_closed=("id", "count"),
                pr_latency_hours_avg=("latency_hours", "mean"),  # raw mean
                pr_merge_ratio=("is_merged", "mean"),               # raw ratio
            )
        )

    # Hard normalize bucket dtype
    opened = _normalize_bucket_column(opened, "month")
    closed = _normalize_bucket_column(closed, "month")

    out = pd.merge(opened, closed, on="month", how="outer").sort_values("month")
    for col in ["prs_opened", "prs_closed", "pr_latency_hours_avg", "pr_merge_ratio"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def _releases_metrics(releases_csv: Path) -> pd.DataFrame:
    """Return bucketed release counts using published_at (fallback created_at)."""
    df = _safe_read_csv(releases_csv)
    if df.empty:
        return pd.DataFrame(columns=["month", "releases_per_month"])

    ts = pd.to_datetime(
        df.get("published_at", df.get("created_at")),
        errors="coerce",
        utc=True
    )
    m = _to_bucket(ts)
    out = (
        pd.DataFrame({"month": m})
        .dropna()
        .groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "releases_per_month"})
    )
    out["releases_per_month"] = pd.to_numeric(out["releases_per_month"], errors="coerce").astype(float)
    return out

# ------------------ Public per-repo builder ------------------

def build_monthly_metrics_for_repo(repo: str, repo_path: Path) -> pd.DataFrame:
    """
    Build per-bucket metrics for a repository by outer-joining:
      commits_local.csv, issues.csv, prs.csv, releases.csv
    Column name kept as 'month' for compatibility; for weekly mode it is week-start date.
    """
    commits_csv = repo_path / "commits_local.csv"
    issues_csv = repo_path / "issues.csv"
    prs_csv = repo_path / "prs.csv"
    releases_csv = repo_path / "releases.csv"

    parts = [
        _commits_metrics(commits_csv),
        _issues_metrics(issues_csv),
        _prs_metrics(prs_csv),
        _releases_metrics(releases_csv),
    ]

    norm_parts = []
    for p in parts:
        if p is None or p.empty:
            continue
        if "month" in p.columns:
            p = _normalize_bucket_column(p, "month")
        norm_parts.append(p)

    out = None
    for p in norm_parts:
        out = p if out is None else pd.merge(out, p, on="month", how="outer")

    if out is None or out.empty:
        return pd.DataFrame()

    out = out.sort_values("month").reset_index(drop=True)
    out.insert(0, "repo", repo)

    # Fill count-like metrics with 0 for stability
    count_like = [
        "commits_per_month", "files_changed_per_commit", "issues_opened", "issues_closed",
        "prs_opened", "prs_closed", "releases_per_month", "bug_issues",
        "bug_issues_core_dev", "bug_issues_external",
    ]
    for c in count_like:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    return out

# ------------------ Pre/post labeling & cohort distributions ------------------


def label_pre_post(
    df: pd.DataFrame,
    adoption_ts: Optional[pd.Timestamp],
    gap_buckets: int = 0,  
) -> pd.DataFrame:
    """
    Label rows as 'pre' if month < adoption bucket; else 'post'.

    """
    out = df.copy()

    freq = _period_code()  # "M" or "W-MON"
    if "month" in out.columns:
        out["month"] = pd.to_datetime(out["month"], errors="coerce") \
                          .dt.to_period(freq).dt.to_timestamp()

    if adoption_ts is None or pd.isna(adoption_ts):
        out["phase"] = "pre"
        return out

    a = pd.to_datetime(adoption_ts, errors="coerce", utc=True)
    if pd.isna(a):
        out["phase"] = "pre"
        return out

    adopt_bucket = (
        pd.Series([a]).dt.tz_convert("UTC").dt.tz_localize(None)
        .dt.to_period(freq).dt.to_timestamp().iloc[0]
    )

    # —— label PRE/POST ——
    out["phase"] = np.where(out["month"] < adopt_bucket, "pre", "post")

    # -- optional: drop +/- gap_buckets around adoption using Period ordinals (robust) --
    if gap_buckets and gap_buckets > 0:
        m_per = out["month"].dt.to_period(freq)
        a_per = pd.Period(adopt_bucket, freq=freq)
        # Use ordinals to get integer distances; avoids MonthEnd offsets
        dist = m_per.astype("int64") - a_per.ordinal
        out = out[dist.abs() > gap_buckets].copy()

    return out


def compute_cohort_distributions(df: pd.DataFrame, metrics: List[str], drop_zeros: bool = False) -> pd.DataFrame:
    """
    Flatten detailed data into long-form distributions:
      columns: cohort, phase, repo, metric, value
    """
    if "repo" not in df.columns:
        df = df.copy()
        df["repo"] = ""

    keep_cols = ["cohort", "repo", "month", "phase"] + [m for m in metrics if m in df.columns]
    data = df[keep_cols].copy()

    parts = []
    for m in metrics:
        if m not in data.columns:
            continue
        sub = data[["cohort", "phase", "repo", m]].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[m])
        if drop_zeros:
            sub = sub[sub[m] != 0]
        sub = sub.rename(columns={m: "value"})
        sub["metric"] = m
        parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=["cohort", "phase", "repo", "metric", "value"])
    return pd.concat(parts, ignore_index=True)

def _cliffs_delta(pre: pd.Series, post: pd.Series) -> float:
    """
    Cliff's delta in [-1, 1].
    Computes P(pre > post) - P(pre < post) using value counts (O(k)).
    Returns np.nan if either group is empty.
    """
    a = pd.Series(pre).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    b = pd.Series(post).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return np.nan

    vals = np.union1d(a, b)
    ca = dict(zip(*np.unique(a, return_counts=True)))
    cb = dict(zip(*np.unique(b, return_counts=True)))

    # cumulative counts to avoid O(n*m)
    pairs_greater = 0
    pairs_less = 0
    cum_a = 0
    cum_b = 0
    for v in vals:
        na = ca.get(v, 0)
        nb = cb.get(v, 0)
        pairs_greater += na * cum_b  # pre==v with post < v
        pairs_less    += nb * cum_a  # post==v with pre < v
        cum_a += na
        cum_b += nb

    return (pairs_greater - pairs_less) / float(n * m)


def summarize_pre_post(dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary per metric:
      pre/post mean, median, std, n
      delta_mean, delta_median, relative_change_mean
      repo_count
      cohens_d  (standardized mean difference using pooled SD)
    """
    cols = [
        "metric", "pre_mean", "post_mean", "pre_median", "post_median",
        "pre_std", "post_std", "n_pre", "n_post",
        "delta_mean", "delta_median", "relative_change_mean",
        "repo_count",
        "cohens_d",
        "cliffs_delta",
        "wilcoxon_W",         
        "wilcoxon_p",
    ]
    if dist_df.empty:
        return pd.DataFrame(columns=cols)

    has_repo = "repo" in dist_df.columns
    rows = []

    for metric in sorted(dist_df["metric"].unique()):
        mdf = dist_df[dist_df["metric"] == metric]

        pre = mdf[mdf["phase"] == "pre"]["value"]
        post = mdf[mdf["phase"] == "post"]["value"]

        # basic stats
        pre_mean    = pre.mean()
        post_mean   = post.mean()
        pre_median  = pre.median()
        post_median = post.median()

        n_pre  = int(pre.shape[0])
        n_post = int(post.shape[0])

        pre_std  = pre.std(ddof=1) if n_pre  > 1 else np.nan
        post_std = post.std(ddof=1) if n_post > 1 else np.nan

        # repo count (unique repos contributing non-null values)
        if has_repo:
            repo_count = int(mdf.loc[mdf["value"].notna(), "repo"].nunique())
        else:
            repo_count = np.nan

        # deltas
        delta_mean   = (post_mean - pre_mean) if (pd.notna(pre_mean) and pd.notna(post_mean)) else np.nan
        delta_median = (post_median - pre_median) if (pd.notna(pre_median) and pd.notna(post_median)) else np.nan
        relative_change_mean = np.nan if (pre_mean is None or pd.isna(pre_mean) or pre_mean == 0) \
                               else (post_mean - pre_mean) / pre_mean

        # --- NEW: Cohen's d (independent groups; pooled SD) ---
        # only compute when both groups have n>=2 and finite SDs
        if (n_pre >= 2) and (n_post >= 2) and pd.notna(pre_std) and pd.notna(post_std):
            df_pool = (n_pre + n_post - 2)
            if df_pool > 0:
                s_pooled_sq = ((n_pre - 1) * (pre_std ** 2) + (n_post - 1) * (post_std ** 2)) / df_pool
                s_pooled = np.sqrt(s_pooled_sq) if s_pooled_sq > 0 else np.nan
            else:
                s_pooled = np.nan
            cohens_d = (post_mean - pre_mean) / s_pooled if (pd.notna(s_pooled) and s_pooled > 0) else np.nan
        else:
            cohens_d = np.nan
        # ----------------------------------------------
        cdelta = _cliffs_delta(pre, post)

        # ---------- Wilcoxon signed-rank (paired by repo) ----------
        wilcoxon_W = np.nan
        wilcoxon_p = np.nan
        try:
            if has_repo:
                pre_by_repo  = mdf[mdf["phase"] == "pre"].groupby("repo")["value"].median()
                post_by_repo = mdf[mdf["phase"] == "post"].groupby("repo")["value"].median()
                paired = pd.concat([pre_by_repo, post_by_repo], axis=1, keys=["pre", "post"]).dropna()

                if len(paired) >= 2: 
                    stat = wilcoxon(
                        paired["post"].to_numpy(),
                        paired["pre"].to_numpy(),
                        alternative="two-sided",
                        zero_method="pratt",
                        nan_policy="omit",
                        method="auto",
                    )
                    wilcoxon_W = float(stat.statistic)
                    wilcoxon_p = float(stat.pvalue)
        except Exception:
            pass

        rows.append({
            "metric": metric,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "pre_median": pre_median,
            "post_median": post_median,
            "pre_std": pre_std,
            "post_std": post_std,
            "n_pre": n_pre,
            "n_post": n_post,
            "delta_mean": delta_mean,
            "delta_median": delta_median,
            "relative_change_mean": relative_change_mean,
            "repo_count": repo_count,
            "cohens_d": cohens_d,  # NEW
            "cliffs_delta": cdelta,  # NEW
            "wilcoxon_W": wilcoxon_W,         # NEW
            "wilcoxon_p": wilcoxon_p,         # NEW
        })

    return pd.DataFrame(rows, columns=cols)

# ------------------ Plotting ------------------

def _pretty_metric_label(metric_key: str) -> str:
    """Human-friendly labels for charts."""
    mapping = {
        "commits_per_month": "Commits / " + _period_label(),
        "files_changed_per_commit": "Files Changed / " + _period_label(),
        "avg_churn_per_commit": "Churn / Commit",
        "issues_opened": "Issues Opened / " + _period_label(),
        "issues_closed": "Issues Closed / " + _period_label(),
        "prs_opened": "PRs Opened / " + _period_label(),
        "prs_closed": "PRs Closed / " + _period_label(),
        "pr_latency_hours_avg": "PR Latency (hrs, avg)",
        "pr_merge_ratio": "Merge Ratio (merged/closed)",
        "merge_ratio": "Merge Ratio (merged/closed)",
        "releases_per_month": "Releases / " + _period_label(),
        "bug_issues": "Bug Issues / " + _period_label(),
        "bug_issues_core_dev": "Bug Issues (Core Dev) / " + _period_label(),
        "bug_issues_external": "Bug Issues (External) / " + _period_label(),
    }
    return mapping.get(metric_key, metric_key)


def plot_distributions_pdf(
    detailed_df: pd.DataFrame,
    metrics: List[str],
    output_path: Path,
    *,
    # log scale controls
    use_log_scale: bool = False,
    log_metrics: Optional[set] = None,      # which metrics to log-scale; None -> default heavy-tailed set
    log_epsilon: float = 1.0,               # substitute for non-positive values when using log
    # winsorization controls
    winsorize: Optional[tuple] = None,      # (lower_q, upper_q); None -> disable
    winsorize_per_metric: Optional[dict] = None,  # per-metric overrides
    # layout
    share_y: bool = True,                   # keep same y-limits across cohorts for a metric
    show_median_labels: bool = True,        # annotate median values and draw lines
    compact: bool = True     ,
                   drop_zeros: bool = False,               # smaller/tighter layout
):
    """
    Create a PDF: for each metric, compare Pre vs Post distributions across cohorts.
    Values are not rounded; only on-figure labels show two decimals.
    """
    print(f"Plot config - log_scale: {use_log_scale}, winsorize: {winsorize}, share_y: {share_y}, compact: {compact}, drop_zeros: {drop_zeros}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Default set of heavy-tailed metrics for optional log-scale
    default_log_metrics = {
        "commits_per_month",
        "files_changed_per_commit",
        "avg_churn_per_commit",
        "issues_opened",
        "issues_closed",
        "prs_opened",
        "prs_closed",
        "releases_per_month",
    }
    if log_metrics is None:
        log_metrics = default_log_metrics

    def _apply_winsorize(arr: np.ndarray, metric: str) -> np.ndarray:
        """Winsorize values with global or per-metric quantiles."""
        if arr.size == 0:
            return arr
        q = None
        if winsorize_per_metric and metric in winsorize_per_metric:
            q = winsorize_per_metric[metric]
        elif winsorize is not None:
            q = winsorize
        if not q:
            return arr
        lo_q, hi_q = q
        lo = np.nanquantile(arr, lo_q)
        hi = np.nanquantile(arr, hi_q)
        return np.clip(arr, lo, hi)

    def _repo_count(df_detail: pd.DataFrame, cohort: str, phase: str, metric: str) -> int:
        """Count unique repos contributing non-null values for a cohort/phase/metric."""
        if metric not in df_detail.columns:
            return 0
        m = (
            (df_detail.get("cohort") == cohort)
            & (df_detail.get("phase") == phase)
            & df_detail[metric].replace([np.inf, -np.inf], np.nan).notna()
        )
        return int(df_detail.loc[m, "repo"].nunique())

    dist = compute_cohort_distributions(detailed_df, metrics=metrics, drop_zeros=drop_zeros)
    if dist.empty:
        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(7, 3.5))
            plt.text(0.5, 0.5, "No data available for distributions.", ha="center", va="center")
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        return

    # Order cohorts for side-by-side comparison
    pref_order = ["overall_base", "overall_quota"]
    cohorts = [c for c in pref_order if c in dist["cohort"].unique()]
    if not cohorts:
        cohorts = sorted(dist["cohort"].unique())

    with PdfPages(output_path) as pdf:
        for metric in metrics:
            mdf = dist[dist["metric"] == metric]
            if mdf.empty:
                fig = plt.figure(figsize=(5.2, 3.4))
                plt.title(_pretty_metric_label(metric))
                plt.text(0.5, 0.5, "No data", ha="center", va="center")
                plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue
            
            # Figure size (compact vs spacious)
            n_cols = len(cohorts)
            fig_w = 4.0 * n_cols if compact else 6.0 * n_cols
            fig_h = 3.2 if compact else 4.8
            fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
            
            # fig.suptitle(_pretty_metric_label(metric), fontsize=12 if compact else 14, y=0.96)

            # Prepare per-axis data + global y-range if share_y
            per_axis = []
            all_vals_for_ylim = []

            for cohort in cohorts:
                cdf = mdf[mdf["cohort"] == cohort]
                pre_vals = cdf[cdf["phase"] == "pre"]["value"].replace([np.inf, -np.inf], np.nan).dropna().values
                post_vals = cdf[cdf["phase"] == "post"]["value"].replace([np.inf, -np.inf], np.nan).dropna().values

                # Optional winsorization
                pre_vals = _apply_winsorize(pre_vals, metric)
                post_vals = _apply_winsorize(post_vals, metric)

                # Optional log-scale (for plotting only)
                apply_log = use_log_scale and (metric in log_metrics)
                if apply_log:
                    pre_vals = np.where(pre_vals <= 0, log_epsilon, pre_vals)
                    post_vals = np.where(post_vals <= 0, log_epsilon, post_vals)

                per_axis.append({"cohort": cohort, "pre": pre_vals, "post": post_vals, "apply_log": apply_log})

                if pre_vals.size:
                    all_vals_for_ylim.append(pre_vals)
                if post_vals.size:
                    all_vals_for_ylim.append(post_vals)

            # Shared y-limits across cohorts
            y_min, y_max = None, None
            if share_y and len(all_vals_for_ylim):
                concat_vals = np.concatenate(all_vals_for_ylim)
                finite_vals = concat_vals[np.isfinite(concat_vals)]
                if finite_vals.size:
                    y_min = np.nanmin(finite_vals)
                    y_max = np.nanmax(finite_vals)
                    pad = 0.05 * (y_max - y_min) if (y_max > y_min) else (0.05 * y_max if y_max > 0 else 1.0)
                    y_min, y_max = max(0, y_min - pad), y_max + pad

            # Draw panels
            for j, item in enumerate(per_axis):
                ax = axes[0, j]
                pre_vals, post_vals = item["pre"], item["post"]
                apply_log = item["apply_log"]

                if pre_vals.size == 0 and post_vals.size == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    ax.set_axis_off()
                    continue

                data = [pre_vals, post_vals]

                # Violin for shape
                parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_alpha(0.25)

                # Box for median/IQR (no rounding)
                ax.boxplot(
                    data,
                    widths=0.18 if compact else 0.2,
                    vert=True,
                    showfliers=False,
                    medianprops=dict(linewidth=1.4),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0),
                    boxprops=dict(linewidth=1.0)
                )

                ax.set_xticks([1, 2])
                ax.set_xticklabels(["Pre-CI", "Post-CI"], fontsize=9 if compact else 10)
                ax.set_ylabel("Value", fontsize=9 if compact else 10)
                ax.set_title(item["cohort"], fontsize=11 if compact else 12)
                ax.grid(True, axis="y", linestyle="--", alpha=0.35)

                if apply_log:
                    ax.set_yscale("log")

                if share_y and (y_min is not None and y_max is not None) and np.isfinite(y_min) and np.isfinite(y_max):
                    ax.set_ylim(y_min, y_max)

                # Annotate n and unique repos
                ymax = ax.get_ylim()[1]
                pre_repos = _repo_count(detailed_df, item["cohort"], "pre", metric)
                post_repos = _repo_count(detailed_df, item["cohort"], "post", metric)
                ax.text(1, ymax, f"n={len(pre_vals)} / repos={pre_repos}", ha="center", va="bottom", fontsize=8)
                ax.text(2, ymax, f"n={len(post_vals)} / repos={post_repos}", ha="center", va="bottom", fontsize=8)

                # Median labels with two-decimal display (data remains full precision)
                if show_median_labels:
                    pre_med = float(np.median(pre_vals)) if pre_vals.size else np.nan
                    post_med = float(np.median(post_vals)) if post_vals.size else np.nan
                    if np.isfinite(pre_med):
                        ax.hlines(pre_med, 0.7, 1.3, colors="tab:orange", linestyles="-", linewidth=1.6, alpha=0.9)
                        ax.text(0.7, pre_med, f" median={pre_med:.2f}", va="center", ha="left", fontsize=8)
                    if np.isfinite(post_med):
                        ax.hlines(post_med, 1.7, 2.3, colors="tab:orange", linestyles="-", linewidth=1.6, alpha=0.9)
                        ax.text(1.7, post_med, f" median={post_med:.2f}", va="center", ha="left", fontsize=8)

            # Tight layout and save page
            if compact:
                fig.tight_layout(rect=[0, 0.01, 1, 0.94], w_pad=0.6)
            else:
                fig.tight_layout(rect=[0, 0.02, 1, 0.95], w_pad=1.0)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)




from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def plot_distributions_pdf3(
    detailed_df: pd.DataFrame,
    metrics: List[str],
    output_path: Path,
    *,
    use_log_scale: bool = False,                     # global switch: prefer log x-axis
    winsorize: Optional[Tuple[float, float]] = None, # (low_q, high_q); None -> off
    drop_zeros: bool = False,                        # drop exact zeros before plotting
    density: bool = True,                            # histogram density vs counts
    bins: int = 40,                                  # default bin count (non-count metrics on linear)
    figsize: Tuple[float, float] = (12.0, 4.6),
    debug: bool = False,                             # page size per metric
    count_metrics: Optional[set] = None,             # integer-like metrics (use integer bins on linear)
    jitter: float = 0.0,                             # small noise for linear+count metrics
    # metrics that must NOT use log even if use_log_scale=True
    log_exempt: Optional[set] = None,

    # ---- NEW: Δ annotation controls (default off to keep behavior unchanged) ----
    show_delta: bool = True,                        # draw Δ (quota - base) for each panel
    delta_stat: str = "median",                      # "median" or "mean" for Δ
) -> None:
    """
    PRE/POST side-by-side; overlay base vs quota histograms with median lines.
    Optional: annotate Δ = quota - base (median/mean) per panel with a double arrow.
    """

    # default count-like set
    if count_metrics is None:
        count_metrics = {
            "commits_per_month", "files_changed_per_commit", "issues_opened", "issues_closed",
            "prs_opened", "prs_closed",
            "pr_merge_ratio", "releases_per_month","merge_ratio",
            "bug_issues", "bug_issues_core_dev", "bug_issues_external",
        }

    # default no-log set (kept linear)
    if log_exempt is None:
        log_exempt = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build long-form once (keeps zero filtering consistent)
    dist = compute_cohort_distributions(detailed_df, metrics, drop_zeros=drop_zeros)
    if dist.empty:
        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, "No data to plot.", ha="center", va="center")
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        return

    # ------- helpers -------
    def _winsor(a: np.ndarray) -> np.ndarray:
        """Clip values to global quantiles if winsorize is provided."""
        if winsorize is None or a.size == 0:
            return a
        lo_q, hi_q = winsorize
        lo = np.nanquantile(a, lo_q)
        hi = np.nanquantile(a, hi_q)
        return np.clip(a, lo, hi)

    def _stat(arr: np.ndarray) -> float:
        """Return median or mean for Δ computation."""
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.nanmedian(arr) if delta_stat.lower() == "median" else np.nanmean(arr))

    with PdfPages(output_path) as pdf:
        for metric in metrics:
            mdf = dist[dist["metric"] == metric]
            if mdf.empty:
                fig = plt.figure(figsize=(6, 3))
                plt.title(metric)
                plt.text(0.5, 0.5, "No metric data", ha="center", va="center")
                plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue

            # ---- per metric repo counts (overall, optional) ----
            if debug and {"repo", "cohort", "value"}.issubset(mdf.columns):
                mask_base_all = mdf["cohort"].astype(str).str.contains("base", case=False, na=False) & mdf["value"].notna()
                mask_quota_all = mdf["cohort"].astype(str).str.contains("quota", case=False, na=False) & mdf["value"].notna()
                n_repo_base_all = int(mdf.loc[mask_base_all, "repo"].nunique())
                n_repo_quota_all = int(mdf.loc[mask_quota_all, "repo"].nunique())
                print(f"[DEBUG][OVERALL] metric={metric} → base repos={n_repo_base_all}, quota repos={n_repo_quota_all}")

            # per-metric log decision
            do_log = bool(use_log_scale and (metric not in log_exempt))
            is_count = metric in count_metrics

            # ---------- collect PRE/POST arrays ----------
            def _vals_for(phase_name: str) -> Tuple[np.ndarray, np.ndarray]:
                """Return (base_vals, quota_vals) for a phase, after winsorization (+ log filter if needed)."""
                subp = mdf[mdf["phase"].astype(str).str.lower() == phase_name]
                a = subp[subp["cohort"].astype(str).str.contains("base",  case=False, na=False)]["value"].to_numpy()
                b = subp[subp["cohort"].astype(str).str.contains("quota", case=False, na=False)]["value"].to_numpy()
                a = _winsor(a); b = _winsor(b)
                if do_log:  # only drop non-positives when actually using log
                    a = a[a > 0]
                    b = b[b > 0]
                return a, b

            base_pre,  quota_pre  = _vals_for("pre")
            base_post, quota_post = _vals_for("post")
            all_x_list = [x for x in (base_pre, quota_pre, base_post, quota_post) if x.size]

            # ---------- choose bins (shared per metric) ----------
            if do_log and is_count and all_x_list:
                # Count metric on log axis: integer edges but log scale
                all_x = np.concatenate(all_x_list)
                vmin = max(1.0, float(np.nanmin(all_x)))  # log requires >0
                vmax = float(np.nanmax(all_x))
                lo = int(np.floor(vmin)); hi = int(np.ceil(vmax))
                cur_bins = np.arange(lo, hi + 1, 1)
            elif do_log and all_x_list:
                # Continuous on log axis: log-spaced bins
                xmin = max(1e-12, float(np.nanmin(np.concatenate(all_x_list))))
                xmax = float(np.nanmax(np.concatenate(all_x_list)))
                if xmax <= xmin:
                    xmax = xmin * 1.1
                cur_bins = np.logspace(np.log10(xmin), np.log10(xmax), bins)
            elif (not do_log) and is_count and all_x_list:
                # Linear + count: integer-centered bins
                all_x = np.concatenate(all_x_list)
                vmin = float(np.nanmin(all_x)); vmax = float(np.nanmax(all_x))
                cur_bins = np.arange(np.floor(vmin) - 0.5, np.ceil(vmax) + 0.5 + 1e-9, 1.0)
            else:
                # Linear + continuous: evenly spaced bins
                if all_x_list:
                    xmin = float(np.nanmin(np.concatenate(all_x_list)))
                    xmax = float(np.nanmax(np.concatenate(all_x_list)))
                    if xmax <= xmin:
                        xmax = xmin + 1.0
                    cur_bins = np.linspace(xmin, xmax, bins)
                else:
                    cur_bins = bins  # matplotlib will handle

            # ---------- figure ----------
            fig, axes = plt.subplots(
                1, 2, figsize=figsize,
                sharex=True, sharey=True,
                gridspec_kw={'wspace': 0.02},
                constrained_layout=True
            )
            # pretty label if you have it; else use raw metric
            try:
                label = _pretty_metric_label(metric)
            except Exception:
                label = metric
            fig.suptitle(label, fontsize=12, fontweight="bold", y=1.02)

            for k, (phase, pair) in enumerate([("pre", (base_pre, quota_pre)),
                                               ("post", (base_post, quota_post))]):
                ax = axes[k]
                base, quota = pair

                # optional jitter only for linear+count (visual smoothing)
                if (not do_log) and is_count and jitter > 0:
                    if base.size:
                        base = base + np.random.uniform(-jitter, jitter, size=base.size)
                    if quota.size:
                        quota = quota + np.random.uniform(-jitter, jitter, size=quota.size)

                if do_log:
                    ax.set_xscale("log")

                # overlapped histograms
                ax.hist(base,  bins=cur_bins, density=density, alpha=0.35,
                        label="base",  color="#1f77b4", edgecolor="none")
                ax.hist(quota, bins=cur_bins, density=density, alpha=0.35,
                        label="quota", color="#ff7f0e", edgecolor="none")

                # medians (kept as before)
                b_med = float(np.nanmedian(base))  if base.size  else np.nan
                q_med = float(np.nanmedian(quota)) if quota.size else np.nan
                if np.isfinite(b_med):
                    ax.axvline(b_med, color="#1f77b4", linewidth=1.6, linestyle="-", label="base median")
                if np.isfinite(q_med):
                    ax.axvline(q_med, color="#ff7f0e", linewidth=1.6, linestyle="-", label="quota median")

                # ---- NEW: Δ annotation (quota - base) using chosen stat ----
                if show_delta:
                    b_stat = _stat(base)
                    q_stat = _stat(quota)
                    if np.isfinite(b_stat) and np.isfinite(q_stat):
                        # place arrow near top of panel
                        y_top = ax.get_ylim()[1]
                        y_anno = y_top * 0.88
                        ax.annotate(
                            "", xy=(q_stat, y_anno), xytext=(b_stat, y_anno),
                            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2, alpha=0.95)
                        )
                        ax.text(
                            (b_stat + q_stat) / 2.0, y_anno,
                            f"Δ={q_stat - b_stat:.2f}",
                            ha="center", va="bottom", fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.75)
                        )

                # cosmetics & legend (right panel only)
                if k == 0:
                    ax.set_ylabel("Density" if density else "Count")
                    ax.spines['right'].set_visible(True)
                    ax.tick_params(axis='y', which='both', right=False, left=True, labelleft=True)
                else:
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)
                    for t in ax.yaxis.get_major_ticks():
                        t.tick1line.set_visible(False)
                        t.tick2line.set_visible(True)
                    for t in ax.yaxis.get_minor_ticks():
                        t.tick1line.set_visible(False)
                        t.tick2line.set_visible(True)
                    ax.yaxis.set_ticks_position('right')
                    ax.spines['right'].set_visible(True)

                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

                # ---- existing per-phase debug (kept) ----
                if debug and "repo" in detailed_df.columns:
                    mask_base = (
                        detailed_df["phase"].astype(str).str.lower().eq(phase)
                        & detailed_df["cohort"].astype(str).str.contains("base", case=False, na=False)
                        & detailed_df[metric].replace([np.inf, -np.inf], np.nan).notna()
                    )
                    mask_quota = (
                        detailed_df["phase"].astype(str).str.lower().eq(phase)
                        & detailed_df["cohort"].astype(str).str.contains("quota", case=False, na=False)
                        & detailed_df[metric].replace([np.inf, -np.inf], np.nan).notna()
                    )
                    nb = int(detailed_df.loc[mask_base, "repo"].nunique())
                    nq = int(detailed_df.loc[mask_quota, "repo"].nunique())
                    print(f"[DEBUG][{phase.upper()}] metric={metric} → base repos={nb}, quota repos={nq}")

            fig.tight_layout(rect=[0, 0.02, 1, 0.98])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)



from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def plot_distributions_box(
    detailed_df: pd.DataFrame,
    metrics: List[str],
    output_path: Path,
    *,
    # which cohorts to show in pairs on the same page-row
    cohort_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    # drop exact zeros before plotting
    drop_zeros: bool = True,
    # y-axis log scale for heavy-tailed metrics
    use_log_y: bool = True,
    # winsorization (clip by quantiles); None → disable
    winsorize: Optional[Tuple[float, float]] = (0.01, 0.99),
    # which statistic to connect & annotate ("median" or "mean")
    delta_stat: str = "median",
    # ----- figure size control -----
    fig_size: Optional[Tuple[float, float]] = None,
    height_per_row: float = 3.6,
    min_width: float = 8.0,
    # ----- spacing control -----
    within_pair_gap: float = 0.25,
    between_cohort_gap: float = 1.2,
    box_width: float = 0.35,
    # colors per cohort (fallback to matplotlib defaults when missing)
    cohort_colors: Optional[Dict[str, str]] = None,
    # label mapper for prettier titles
    pretty_metric_label_fn=None,  # e.g. pass _pretty_metric_label; defaults to identity

    # ====== NEW: show inner distribution shapes ======
    show_violin: bool = True,        # overlay light KDE violins behind boxes
    violin_alpha: float = 0.18,       # transparency for violins
    show_points: bool = False,        # overlay jittered points (downsampled)
    max_points: Optional[int] = 1500, # cap total points per subplot to avoid clutter
    point_alpha: float = 0.35,
    point_size: float = 8.0,
    point_jitter: float = 0.04,       # x-jitter around each box position
    random_state: Optional[int] = 0,  # for reproducible downsampling/jitter
    notch: bool = False,              # notch boxes (median CI)
) -> None:
    """
    For each metric (one PDF page):
      - Draw PRE and POST boxplots for each cohort (e.g., base & quota).
      - Optionally overlay violins (KDE) and/or jittered points to show within-bin shapes.
      - Connect the PRE→POST statistic (median/mean) with a line and annotate Δ.
      - Layout/spacing unchanged unless *show_violin/show_points* are enabled.
    """

    # Defaults
    if cohort_pairs is None:
        cohort_pairs = [("overall_base", "overall_quota")]
    if pretty_metric_label_fn is None:
        pretty_metric_label_fn = (lambda m: m)
    rng = np.random.default_rng(random_state)

    # Build long-form once
    from services.rq2_helper import compute_cohort_distributions  # reuse util
    dist = compute_cohort_distributions(detailed_df, metrics, drop_zeros=drop_zeros)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dist.empty:
        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, "No data to plot.", ha="center", va="center")
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        return

    # clean types
    dist["cohort"] = dist["cohort"].astype(str)
    dist["phase"]  = dist["phase"].astype(str).str.lower()

    # winsor helper
    def _winsor(a: np.ndarray) -> np.ndarray:
        if winsorize is None or a.size == 0:
            return a
        lo_q, hi_q = winsorize
        lo = np.nanquantile(a, lo_q); hi = np.nanquantile(a, hi_q)
        return np.clip(a, lo, hi)

    # stat selector
    def _stat(vals: np.ndarray) -> float:
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan
        return np.median(vals) if delta_stat == "median" else np.mean(vals)

    with PdfPages(output_path) as pdf:
        for metric in metrics:
            mdf = dist[dist["metric"] == metric].copy()
            if mdf.empty:
                fig = plt.figure(figsize=(6, 3))
                plt.title("")
                plt.text(0.5, 0.5, "No metric data", ha="center", va="center")
                plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
                continue
             # ====== NEW: 打印 repo 数 ======
            repos_base = mdf.loc[mdf["cohort"].str.contains("base", case=False), "repo"].nunique()
            repos_quota = mdf.loc[mdf["cohort"].str.contains("quota", case=False), "repo"].nunique()
            print(f"[INFO] metric={metric} → base repos={repos_base}, quota repos={repos_quota}")
                
            # --- figure size ---
            n_rows = len(cohort_pairs)
            if fig_size is not None:
                fig_w, fig_h = fig_size
            else:
                fig_w = min_width
                fig_h = max(3.0, n_rows * height_per_row)

            fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)
            fig.suptitle(
                pretty_metric_label_fn(metric),
                fontsize=13, fontweight="bold",
                x=0.55, y=0.89, ha="center"
            )

            for r, (c1, c2) in enumerate(cohort_pairs):
                ax = axes[r, 0]

                # gather arrays
                data_blocks = []  # [(label, np.array, color)]
                for cohort in (c1, c2):
                    sub = mdf[mdf["cohort"] == cohort]
                    pre_vals  = _winsor(sub[sub["phase"] == "pre"]["value"].to_numpy())
                    post_vals = _winsor(sub[sub["phase"] == "post"]["value"].to_numpy())
                    color = cohort_colors.get(cohort) if cohort_colors else None
                    data_blocks.append((f"{cohort}\nPRE",  pre_vals,  color))
                    data_blocks.append((f"{cohort}\nPOST", post_vals, color))

                # --- positions: tighter PRE/POST within the same cohort ---
                x0 = 1.0
                x1 = x0 + within_pair_gap
                x2 = x1 + between_cohort_gap
                x3 = x2 + within_pair_gap
                positions = np.array([x0, x1, x2, x3], dtype=float)

                # labels shown only as PRE/POST
                labels = ["PRE", "POST", "PRE", "POST"]
                arrays = [db[1] for db in data_blocks]

                # ====== (A) optional violins: show full distribution shape ======
                if show_violin:
                    vparts = ax.violinplot(
                        arrays,
                        positions=positions,
                        widths=box_width * 1.6,
                        showmeans=False, showmedians=False, showextrema=False
                    )
                    for i, body in enumerate(vparts["bodies"]):
                        col = data_blocks[i][2]
                        if col is None:  # fallback colors
                            body.set_facecolor("C0" if i < 2 else "C1")
                        else:
                            body.set_facecolor(col)
                        body.set_alpha(violin_alpha)
                        body.set_edgecolor("none")

                # ====== (B) main boxes ======
                bp = ax.boxplot(
                    arrays,
                    positions=positions,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    notch=notch,
                )

                # colorize boxes
                for i, patch in enumerate(bp['boxes']):
                    color = data_blocks[i][2]
                    if color:
                        patch.set_facecolor(color)
                        patch.set_alpha(0.22)

                # ====== (C) optional jitter points ======
                if show_points:
                    # total points per subplot capped -> equal quota per box
                    per_box_cap = None
                    total_pts = sum(len(a) for a in arrays)
                    if (max_points is not None) and (total_pts > max_points):
                        per_box_cap = max(1, max_points // max(1, len(arrays)))

                    for i, arr in enumerate(arrays):
                        if arr.size == 0:
                            continue
                        x0 = np.full(arr.shape[0], positions[i], dtype=float)
                        # downsample
                        if per_box_cap is not None and arr.size > per_box_cap:
                            idx = rng.choice(arr.size, size=per_box_cap, replace=False)
                            arr = arr[idx]
                            x0 = x0[idx]
                        # jitter
                        x0 = x0 + rng.uniform(-point_jitter, point_jitter, size=arr.size)
                        ax.scatter(x0, arr, s=point_size, alpha=point_alpha, color="black", linewidths=0, zorder=3)

                # connect PRE→POST statistic lines & annotate Δ（带双向箭头 + 百分比，适配对数y轴）
                for i, cohort in enumerate((c1, c2)):
                    x_pre  = positions[i*2]
                    x_post = positions[i*2 + 1]
                    y_pre  = _stat(arrays[i*2])
                    y_post = _stat(arrays[i*2 + 1])

                    if not (np.isfinite(y_pre) and np.isfinite(y_post)):
                        continue

                    # connecting line
                    ax.plot(
                        [x_pre, x_post], [y_pre, y_post],
                        color=(cohort_colors.get(cohort, 'tab:blue') if cohort_colors else 'tab:blue'),
                        linewidth=1.8, alpha=0.95
                    )

                    # absolute and relative delta
                    delta = y_post - y_pre
                    rel = (delta / y_pre) if (y_pre != 0 and np.isfinite(y_pre)) else np.nan

                    # color: green/red/gray for positive/negative/zero delta
                    tcolor = "#2ca02c" if delta > 0 else ("#d62728" if delta < 0 else "0.3")

                    # arrow
                    # ax.annotate(
                    #     "", xy=(x_post, 0.90), xytext=(x_pre, 0.90),
                    #     xycoords=("data", "axes fraction"),
                    #     textcoords=("data", "axes fraction"),
                    #     arrowprops=dict(arrowstyle="<->", lw=1.4, color="0.35", alpha=0.95)
                    # )

                    # text
                    label = f"Δ={delta:.2f}" + (f" ({rel*100:+.1f}%)" if np.isfinite(rel) else "")
                    ax.text(
                        (x_pre + x_post) / 2.0, 0.90, label,
                        transform=ax.get_xaxis_transform(), 
                        ha="center", va="bottom",
                        fontsize=9, color=tcolor,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.8)
                    )



                # cosmetics
                ax.set_xticks(positions, labels, rotation=0)
                ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                if use_log_y:
                    ax.set_yscale("log")
                    ax.set_ylabel("Value (log)")
                else:
                    ax.set_ylabel("Value")

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)