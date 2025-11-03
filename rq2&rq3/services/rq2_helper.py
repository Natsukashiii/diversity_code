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