# rdd_rq3.py
# -*- coding: utf-8 -*-
"""
RQ3: Run RDD over overall or stratified subdatasets, with optional mixed-effects.
- Loads base/quota CSV as the dataset index.
- Builds monthly metrics per repo from DATASET_INFO_DIR.
- Fits RDD: Y ~ time + intervention + time_after + controls
- Mixed effects: random intercept by repo (groups) and optional random intercept by language (vc).
- Saves full results to RQ3_DATA_PATH.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from scipy.stats import chi2
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information"
)
# ------------------ CONFIG ------------------
from config.load_config import load_config

cfg = load_config()
RQ1_DATA_PATH = Path(str(cfg.path.rq1_path))
RQ3_DATA_PATH = Path(str(cfg.path.rq3_data_path))
DATASET_INFO_DIR = Path(str(cfg.path.dataset_info_path))
STRA_CSV = RQ1_DATA_PATH / "base.csv"
QUOTA_CSV = RQ1_DATA_PATH / "quota.csv"

DEFAULT_OUT_OVERALL = RQ3_DATA_PATH / "rdd_overall.csv"
DEFAULT_OUT_BUCKET  = RQ3_DATA_PATH / "rdd_by_bucket.csv"

# ------------------ METRICS ------------------
METRICS = [
    "commit_frequency",
    "modified_files",
    "code_churn_size",
    "issues_open_count",
    "issues_close_count",
    "prs_open_count",
    "prs_close_count",
    "pr_latency",
    "pr_merge_ratio",
    "commit_merge_ratio",
    "releases_count",
]

def _stars(p: float) -> str:
    if pd.isna(p): return ""
    if p <= 0.001: return "***"
    if p <= 0.01:  return "**"
    if p <= 0.05:  return "*"
    return ""

def _trim_1pct_two_sided(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 10:
        return s  # too small; skip
    lo = np.nanpercentile(s, 1)
    hi = np.nanpercentile(s, 99)
    return s.where((s >= lo) & (s <= hi), np.nan)

def _compute_vif(X_fe: pd.DataFrame) -> pd.DataFrame:
    """VIF on numeric fixed-effects design (exclude intercept)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = X_fe.copy()
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    if X.shape[1] == 0:
        return pd.DataFrame({"term": [], "VIF": []})
    # small jitter avoids perfect collinearity blow-ups
    X = X + np.random.normal(scale=1e-10, size=X.shape)
    vifs = []
    for i in range(X.shape[1]):
        try:
            vifs.append(variance_inflation_factor(X.values, i))
        except Exception:
            vifs.append(np.nan)
    return pd.DataFrame({"term": X.columns.tolist(), "VIF": vifs})

def _type2_wald_chi2(res, term_names):
    """
    Type-II Wald χ² simple approximation for single-column terms.
    """
    out = {}
    cov = res.cov_params()
    coefs = res.params
    for t in term_names:
        if (t in coefs.index) and (t in cov.index) and (t in cov.columns):
            b = np.array([coefs[t]])
            V = np.array([[cov.loc[t, t]]])
            try:
                val = float(b @ np.linalg.inv(V) @ b.T)
            except Exception:
                val = np.nan
            out[t] = val
    return out

def _approx_r2_marginal_conditional_mixed(res, X_fe: pd.DataFrame, df: pd.DataFrame,
                                          have_proj_intercept: bool,
                                          have_proj_slope: bool,
                                          have_lang_vc: bool):
    """
    Nakagawa & Schielzeth simple R²m/R²c approximation for MixedLM.
    """
    try:
        # FE variance
        fe_cols = [c for c in X_fe.columns if c in res.params.index]
        mu = np.asarray(X_fe[fe_cols] @ res.params[fe_cols])
        var_fe = float(np.nanvar(mu, ddof=1))

        sigma2 = float(res.scale)
        var_re = 0.0

        # language vc
        if have_lang_vc and hasattr(res, "vcomp") and res.vcomp is not None:
            try:
                v_lang = float(np.nanmean(res.vcomp))
            except Exception:
                v_lang = 0.0
            var_re += max(0.0, v_lang)

        # project RE
        if hasattr(res, "cov_re") and res.cov_re is not None:
            C = np.asarray(res.cov_re)
            if C.shape == (1, 1) and have_proj_intercept:
                var_re += float(max(0.0, C[0, 0]))
            elif C.shape[0] >= 2 and have_proj_intercept and have_proj_slope:
                var_u0 = float(max(0.0, C[0, 0]))
                var_u1 = float(max(0.0, C[1, 1]))
                cov01  = float(C[0, 1])
                z = df["intervention"].astype(float).values
                var_z = float(np.nanvar(z, ddof=1))
                # Cov(1,z)=0，因此 var(u0 + u1*z) = var_u0 + var_u1*var_z
                var_re += max(0.0, var_u0 + var_u1 * var_z)  
        denom = var_fe + var_re + sigma2
        if denom <= 0 or not np.isfinite(denom):
            return np.nan, np.nan
        return float(var_fe / denom), float((var_fe + var_re) / denom)
    except Exception:
        return np.nan, np.nan


def _parse_iso8601_series(s: pd.Series) -> pd.Series:
    """
    Fast ISO-8601 parser with graceful fallback.
    1) Try strict '%Y-%m-%dT%H:%M:%SZ' (fast).
    2) For failures, fallback to pandas flexible parser (slower).
    Returns tz-aware (UTC) timestamps.
    """
    if s is None:
        return pd.Series(pd.NaT, index=[])
    s = s.astype(str)
    fast = pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%SZ", utc=True, errors="coerce")
    mask = fast.isna()
    if mask.any():
        slow = pd.to_datetime(s[mask], utc=True, errors="coerce")
        fast.loc[mask] = slow
    return fast


# ------------------ Helpers ------------------

def _repo_dir(repo: str) -> Path:
    """Resolve repo folder in DATASET_INFO_DIR; try 'owner/name' then 'owner__name'."""
    d1 = DATASET_INFO_DIR / repo
    if d1.exists():
        return d1
    return DATASET_INFO_DIR / repo.replace("/", "__")

def _month_floor(ts: pd.Series) -> pd.Series:
    """Map timestamps to month-start (UTC-naive)."""
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp(how="start").dt.tz_localize(None)

def _safe_read_csv(path: Path,
                   usecols: Optional[List[str]] = None,
                   dtype_map: Optional[dict] = None,
                   parse_date_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Robust CSV reader:
    - low_memory=False to avoid chunk-wise dtype guessing.
    - For very wide, mixed-type files (prs/gactions/issues), default to dtype=str.
    - Optionally pass a dtype_map/parse_date_cols for fine control.
    """
    if not path.exists():
        return None

    # Heuristic: files that often have mixed types / nested payloads
    name = path.name.lower()
    force_str = any(k in name for k in ["prs.csv", "gactions.csv", "issues.csv", "jobs.csv", "releases.csv"])

    # Prefer caller-provided dtype_map; otherwise force strings for “noisy” files.
    dtype = dtype_map if dtype_map is not None else (str if force_str else None)

    try:
        return pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype,
            parse_dates=parse_date_cols,   # None by default; we parse later where needed
            low_memory=False,              # <— turn off chunked inference (removes DtypeWarning)
            engine="c"
        )
    except Exception:
        # Fallback: read everything as string to succeed, then let downstream coerce.
        try:
            return pd.read_csv(
                path,
                usecols=usecols,
                dtype=str,
                low_memory=False,
                engine="c"
            )
        except Exception:
            return None

def _agg_commits(repo: str) -> pd.DataFrame:
    """Monthly commit metrics: frequency, files mean, churn mean, merge ratio."""
    d = _repo_dir(repo) / "commits_local.csv"
    df = _safe_read_csv(d)
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "commit_frequency", "modified_files",
                                     "code_churn_size", "commit_merge_ratio"])
    df["month"] = _month_floor(df["author_date"])
    df["files_changed"] = pd.to_numeric(df.get("files_changed", 0), errors="coerce")
    df["insertions"]    = pd.to_numeric(df.get("insertions", 0), errors="coerce")
    df["deletions"]     = pd.to_numeric(df.get("deletions", 0), errors="coerce")
    df["is_merge"]      = pd.to_numeric(df.get("is_merge", 0), errors="coerce").fillna(0).astype(int)
    df["churn"]         = df["insertions"].fillna(0) + df["deletions"].fillna(0)

    grp = df.groupby("month", as_index=False).agg(
        commit_frequency=("author_date", "count"),
        modified_files=("files_changed", "mean"),
        code_churn_size=("churn", "mean"),
        merges=("is_merge", "sum"),
        commits=("author_date", "count"),
    )
    grp["commit_merge_ratio"] = np.where(grp["commits"] > 0, grp["merges"] / grp["commits"], np.nan)
    return grp[["month", "commit_frequency", "modified_files", "code_churn_size", "commit_merge_ratio"]]

def _agg_issues(repo: str) -> pd.DataFrame:
    """Monthly open/close issues count."""
    d = _repo_dir(repo) / "issues.csv"
    df = _safe_read_csv(d)
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "issues_open_count", "issues_close_count"])
    df["created_at_ts"] = _parse_iso8601_series(df.get("created_at"))
    df["closed_at_ts"]  = _parse_iso8601_series(df.get("closed_at"))
    df["created_at_m"]  = _month_floor(df["created_at_ts"])
    df["closed_at_m"]   = _month_floor(df["closed_at_ts"])
    open_cnt  = df.groupby("created_at_m").size().reset_index(name="issues_open_count").rename(columns={"created_at_m":"month"})
    close_cnt = df.dropna(subset=["closed_at_m"]).groupby("closed_at_m").size().reset_index(name="issues_close_count").rename(columns={"closed_at_m":"month"})
    out = pd.merge(open_cnt, close_cnt, how="outer", on="month").sort_values("month")
    return out

def _agg_prs(repo: str) -> pd.DataFrame:
    """Monthly PR open/close counts, mean latency (hours), merge ratio among closed."""
    d = _repo_dir(repo) / "prs.csv"
    df = _safe_read_csv(d)
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "prs_open_count", "prs_close_count", "pr_latency", "pr_merge_ratio"])
    df["created_at_ts"] = _parse_iso8601_series(df.get("created_at"))
    df["closed_at_ts"]  = _parse_iso8601_series(df.get("closed_at"))
    df["merged_at_ts"]  = _parse_iso8601_series(df.get("merged_at"))
    df["created_at_m"] = _month_floor(df["created_at_ts"])
    df["closed_at_m"]  = _month_floor(df["closed_at_ts"])

    open_cnt = df.groupby("created_at_m").size().reset_index(name="prs_open_count").rename(columns={"created_at_m":"month"})
    closed   = df.dropna(subset=["closed_at_m"]).copy()
    close_cnt = closed.groupby("closed_at_m").size().reset_index(name="prs_close_count").rename(columns={"closed_at_m":"month"})
    closed["latency_h"] = (closed["closed_at_ts"] - closed["created_at_ts"]).dt.total_seconds() / 3600.0
    latency = closed.groupby("closed_at_m")["latency_h"].mean().reset_index().rename(columns={"closed_at_m":"month","latency_h":"pr_latency"})
    closed["merged_flag"] = closed["merged_at_ts"].notna().astype(int)
    merges = closed.groupby("closed_at_m")["merged_flag"].agg(["sum","count"]).reset_index().rename(columns={"closed_at_m":"month"})
    merges["pr_merge_ratio"] = np.where(merges["count"]>0, merges["sum"]/merges["count"], np.nan)
    merges = merges[["month","pr_merge_ratio"]]

    out = pd.merge(open_cnt, close_cnt, how="outer", on="month")
    out = pd.merge(out, latency, how="outer", on="month")
    out = pd.merge(out, merges, how="outer", on="month")
    return out.sort_values("month")

def _agg_releases(repo: str) -> pd.DataFrame:
    """Monthly releases count."""
    d = _repo_dir(repo) / "releases.csv"
    df = _safe_read_csv(d)
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "releases_count"])
    ts_pub = _parse_iso8601_series(df.get("published_at"))
    ts_cre = _parse_iso8601_series(df.get("created_at"))
    ts = ts_pub.fillna(ts_cre)
    month = ts.dt.to_period("M").dt.to_timestamp(how="start").dt.tz_localize(None)
    grp = pd.DataFrame({"month": month}).dropna().groupby("month").size().reset_index(name="releases_count")
    return grp

def build_monthly_metrics_for_repo(repo: str) -> pd.DataFrame:
    """Return DataFrame: ['month'] + METRICS."""
    parts = [_agg_commits(repo), _agg_issues(repo), _agg_prs(repo), _agg_releases(repo)]
    out = None
    for p in parts:
        if p is None or p.empty: 
            continue
        out = p if out is None else pd.merge(out, p, how="outer", on="month")
    if out is None:
        return pd.DataFrame(columns=["month"] + METRICS)
    out = out.sort_values("month").reset_index(drop=True)
    for m in METRICS:
        if m not in out.columns: out[m] = np.nan
    return out[["month"] + METRICS]

# ------------------ RDD preparation ------------------

def _build_design(df: pd.DataFrame,
                  adoption_ts: pd.Timestamp,
                  skip_transition_month: bool = True) -> pd.DataFrame:
    """Add RDD columns: time, intervention, time_after; optionally drop adoption month."""
    d = df.copy().dropna(subset=["month"]).sort_values("month")
    if d.empty:
        return d

    # normalize to naive timestamps at month start
    d["month"] = pd.to_datetime(d["month"]).dt.tz_localize(None)
    # helper: convert a timestamp series to integer "month index" = year*12 + month
    def _month_index(ts: pd.Series) -> pd.Series:
        ts = pd.to_datetime(ts)
        return ts.dt.year.astype(int) * 12 + ts.dt.month.astype(int)

    # compute month indices
    mi = _month_index(d["month"])
    start_idx = int(mi.min())

    # adoption month index
    adopt_ts = pd.to_datetime(adoption_ts).tz_localize(None)
    adopt_idx = int(_month_index(pd.Series([adopt_ts]))[0])

    # optionally drop the transition (adoption) month
    if skip_transition_month:
        keep = mi != adopt_idx
        d = d.loc[keep].copy()
        mi = mi.loc[keep]

    # time: months since first observed month
    d["time"] = (mi - start_idx).astype(int)

    # intervention: 1 if month >= adoption month
    d["intervention"] = (mi >= adopt_idx).astype(int)

    # time_after: months since adoption (0 at adoption month; if dropped, starts at 1)
    d["time_after"] = 0
    after_mask = mi >= adopt_idx
    d.loc[after_mask, "time_after"] = (mi[after_mask] - adopt_idx).astype(int)

    d["time_centered"] = d["time"] - d["time"][d["intervention"]==0].mean()
    d["post_time_centered"] = d["time_centered"] * d["intervention"]

    return d

# ------------------ Config ------------------

@dataclass
class RDDConfig:
    # model structure
    mixed_effects: bool = True                 # MixedLM vs OLS
    repo_random_intercept: bool = True         # (1|repo)
    repo_random_slope_intervention: bool = True# (intervention|repo)
    language_random_intercept: bool = True     # vc for language
    language_fixed_effects: bool = False       # FE of language (paper uses RE)
    # design
    skip_transition_month: bool = True
    # sample thresholds
    min_obs: int = 8
    # OLS options
    robust_cov: str = "HC1"                    # for OLS
    # optimizer
    max_iter: int = 300
    # controls (fixed covariates)
    add_controls: bool = True
    # columns in index CSV
    col_total_commits: str = "commits"
    col_num_authors: str  = "contributors"
    col_created_at: str   = "createdAt"
    # data hygiene
    apply_log1p_y: bool = False                # enable if you want log(1+y)
    trim_outliers_1pct: bool = True            # trim response per metric
    # VIF guard
    vif_threshold: float = 3.0

# ------------------ Modeling ------------------

def fit_rdd(df: pd.DataFrame,
            metric: str,
            cfg: RDDConfig) -> Dict[str, object]:
    """
    General RDD fit for one metric.
    Returns a rich dict for CSV output (coeffs, se, p, stars, Wald chi2, R2m/R2c, VIF, Ns...).
    """
    out = {
        "metric": metric,
        "n_obs": 0, "n_projects": 0,
        "coef_time": np.nan, "se_time": np.nan, "p_time": np.nan, "stars_time": "",
        "coef_level_gamma": np.nan, "se_level_gamma": np.nan, "p_level_gamma": np.nan, "stars_level_gamma": "",
        "coef_slope_delta": np.nan, "se_slope_delta": np.nan, "p_slope_delta": np.nan, "stars_slope_delta": "",
        "coef_log_total_commits": np.nan, "se_log_total_commits": np.nan, "p_log_total_commits": np.nan, "stars_log_total_commits": "",
        "coef_age_at_travis_months": np.nan, "se_age_at_travis_months": np.nan, "p_age_at_travis_months": np.nan, "stars_age_at_travis_months": "",
        "coef_log_num_authors": np.nan, "se_log_num_authors": np.nan, "p_log_num_authors": np.nan, "stars_log_num_authors": "",
        "wald_time": np.nan, "wald_intervention": np.nan, "wald_time_after": np.nan,
        "r2_marginal": np.nan, "r2_conditional": np.nan,
        "vif_max": np.nan, "vif_terms": "",
    }

    need = [metric, "time", "intervention", "time_after", "repo"]
    d = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    if d.empty:
        return out

    # hygiene: trim 1% + optional log1p
    y_raw = d[metric]
    if cfg.trim_outliers_1pct:
        y_raw = _trim_1pct_two_sided(y_raw)
    if cfg.apply_log1p_y:
        y = np.log1p(y_raw)
    else:
        y = y_raw
    d = d.assign(**{metric: y})
    d = d.dropna(subset=[metric, "time", "intervention", "time_after", "repo"])
    if d.empty:
        return out

    n_projects = int(d["repo"].nunique())
    n_obs = int(d.shape[0])
    out["n_obs"] = n_obs
    out["n_projects"] = n_projects
    if n_obs < cfg.min_obs:
        return out

    # build FE terms
    fe_terms = ["time_centered", "intervention", "time_after_ortho"]
    if cfg.add_controls:
        for c in ["log_total_commits", "age_at_travis_months", "log_num_authors"]:
            if c in d.columns:
                fe_terms.append(c)
    if cfg.language_fixed_effects and "mainLanguage" in d.columns and d["mainLanguage"].nunique() > 1:
        d["mainLanguage"] = d["mainLanguage"].fillna("UNK")
        fe_terms.append("C(mainLanguage)")

    formula = f"{metric} ~ " + " + ".join(fe_terms)

    # ---- orthogonalize time_after against time_centered ----
    if "time_after" in d.columns and "time_centered" in d.columns:
        try:
            Xa = sm.add_constant(d["time_centered"].astype(float))
            ya = d["time_after"].astype(float)
            alpha = sm.OLS(ya, Xa, missing="drop").fit()
            d["time_after_ortho"] = ya - alpha.predict(Xa)
        except Exception:
            d["time_after_ortho"] = d.get("time_after", np.nan)
    else:
        d["time_after_ortho"] = d.get("time_after", np.nan)

    # Optional: mean-centering for extra numerical stability
    d["time_centered"]    = d["time_centered"] - d["time_centered"].mean()
    d["time_after_ortho"] = d["time_after_ortho"] - d["time_after_ortho"].mean()

    # --- Prepare FE design for VIF / R² (independent of model flavor) ---
    try:
        y_fe, X_fe = dmatrices(formula, data=d, return_type="dataframe")
        X_fe.rename(columns={c: "Intercept" if c == "Intercept" or c == "Intercept" else c for c in X_fe.columns},
                    inplace=True)
        vif_df = _compute_vif(X_fe)
        out["vif_max"] = float(vif_df["VIF"].max()) if not vif_df.empty else np.nan
        out["vif_terms"] = ";".join([f"{r.term}:{round(float(r.VIF),3)}" for _, r in vif_df.iterrows()]) if not vif_df.empty else ""
    except Exception:
        X_fe = None  # still proceed

    # --- Mixed effects ---
    if cfg.mixed_effects:
        try:
            # groups = repo; RE: intercept +/- slope(intervention)
            re_formula = "1"
            if cfg.repo_random_slope_intervention:
                re_formula = "1 + intervention"

            # language VC
            vc = None
            if cfg.language_random_intercept and "mainLanguage" in d.columns and d["mainLanguage"].nunique() > 1:
                vc = {"language": "0 + C(mainLanguage)"}

            # Build MixedLM with patsy: we use data d and the same FE formula
            model = sm.MixedLM.from_formula(formula=formula,
                                            groups=d["repo"],
                                            re_formula=re_formula,
                                            vc_formula=vc,
                                            data=d,
                                            missing="drop")
            res = model.fit(reml=True, method="lbfgs", maxiter=cfg.max_iter, disp=False)

            # Pull stats
            params = res.params
            bse    = res.bse
            pvals  = res.pvalues

            for nm_src, nm_dst_base in [
                ("time_centered", "time"),
                ("intervention", "level_gamma"),
                ("time_after_ortho", "slope_delta"),
            ]:
                if nm_src in params.index:
                    out[f"coef_{nm_dst_base}"] = float(params[nm_src])
                    out[f"se_{nm_dst_base}"]   = float(bse.get(nm_src, np.nan))
                    p = float(pvals.get(nm_src, np.nan))
                    out[f"p_{nm_dst_base}"]    = p
                    out[f"stars_{nm_dst_base}"]= _stars(p)

            # controls
            for nm in ["log_total_commits", "age_at_travis_months", "log_num_authors"]:
                if nm in params.index:
                    out[f"coef_{nm}"]  = float(params[nm])
                    out[f"se_{nm}"]    = float(bse.get(nm, np.nan))
                    p = float(pvals.get(nm, np.nan))
                    out[f"p_{nm}"]     = p
                    out[f"stars_{nm}"] = _stars(p)

            # Type-II Wald χ²
            wald = _type2_wald_chi2(res, ["time_centered","intervention","time_after_ortho"])
            out["wald_time"]         = wald.get("time_centered", np.nan)
            out["wald_intervention"] = wald.get("intervention", np.nan)
            out["wald_time_after"]   = wald.get("time_after_ortho", np.nan)

            # R2m / R2c (approx)
            if X_fe is not None:
                r2m, r2c = _approx_r2_marginal_conditional_mixed(
                    res, X_fe, d,
                    have_proj_intercept=cfg.repo_random_intercept,
                    have_proj_slope=cfg.repo_random_slope_intervention,
                    have_lang_vc=cfg.language_random_intercept
                )
                out["r2_marginal"]    = r2m
                out["r2_conditional"] = r2c

        except Exception:
            # keep NaNs
            pass
        return out

    # --- OLS (robust/cluster SE) ---
    try:
        y, X = dmatrices(formula, data=d, return_type="dataframe")
        ols_model = sm.OLS(y, X, missing="drop")
        if cfg.robust_cov and cfg.robust_cov.lower() == "cluster" and "repo" in d.columns:
            res = ols_model.fit(cov_type="cluster", cov_kwds={"groups": d["repo"]})
        else:
            res = ols_model.fit(cov_type=(cfg.robust_cov or "nonrobust"))

        params = res.params
        bse    = res.bse
        pvals  = res.pvalues

        for nm_src, nm_dst_base in [
            ("time_centered", "time"),
            ("intervention", "level_gamma"),
            ("time_after_ortho", "slope_delta"),
        ]:
            if nm_src in params.index:
                out[f"coef_{nm_dst_base}"] = float(params[nm_src])
                out[f"se_{nm_dst_base}"]   = float(bse.get(nm_src, np.nan))
                p = float(pvals.get(nm_src, np.nan))
                out[f"p_{nm_dst_base}"]    = p
                out[f"stars_{nm_dst_base}"]= _stars(p)

        for nm in ["log_total_commits", "age_at_travis_months", "log_num_authors"]:
            if nm in params.index:
                out[f"coef_{nm}"]  = float(params[nm])
                out[f"se_{nm}"]    = float(bse.get(nm, np.nan))
                p = float(pvals.get(nm, np.nan))
                out[f"p_{nm}"]     = p
                out[f"stars_{nm}"] = _stars(p)

        # Type-II Wald χ²
        wald = _type2_wald_chi2(res, ["time_centered", "intervention", "time_after_ortho"])
        out["wald_time"]         = wald.get("time_centered", np.nan)
        out["wald_intervention"] = wald.get("intervention", np.nan)
        out["wald_time_after"]   = wald.get("time_after_ortho", np.nan)

        # OLS: R2m=R2c=R2
        try:
            out["r2_marginal"] = float(res.rsquared)
            out["r2_conditional"] = float(res.rsquared)
        except Exception:
            pass

    except Exception:
        # keep NaNs
        pass

    return out


# ------------------ Dataset IO & panel ------------------

def _days_to_months(days):
    """Vectorized: days -> months. Works for scalar, Series, or ndarray."""
    if isinstance(days, (pd.Series, np.ndarray)):
        return pd.to_numeric(days, errors="coerce") / 30
    # scalar
    try:
        return float(days) / 30
    except Exception:
        return np.nan

def load_index(csv_path: Path, cfg: RDDConfig) -> pd.DataFrame:
    """
    Load dataset index.
    Expected columns (best effort): repo, mainLanguage, ci_adoption_time, bucket_id, bucket_name,
    createdAt, commits, contributors
    """
    usecols = ["repo", "mainLanguage", "ci_adoption_time", "bucket_id", "bucket_name",
               cfg.col_created_at, cfg.col_total_commits, cfg.col_num_authors]
    df = _safe_read_csv(csv_path, usecols=None)  # read all to be robust
    if df is None or df.empty:
        raise FileNotFoundError(f"Failed to load dataset index: {csv_path}")

    # Normalize essential columns
    for col in ["repo", "mainLanguage", "ci_adoption_time", "bucket_id", "bucket_name",
                cfg.col_created_at, cfg.col_total_commits, cfg.col_num_authors]:
        if col not in df.columns:
            df[col] = np.nan

    df["ci_adoption_time"] = _parse_iso8601_series(df.get("ci_adoption_time")).dt.tz_convert(None)
    df[cfg.col_created_at] = _parse_iso8601_series(df.get(cfg.col_created_at)).dt.tz_convert(None)

    # controls (project-level)
    # log_total_commits = log1p(commits)
    df["log_total_commits"] = np.log1p(pd.to_numeric(df[cfg.col_total_commits], errors="coerce"))
    # log_num_authors = log1p(contributors)
    df["log_num_authors"] = np.log1p(pd.to_numeric(df[cfg.col_num_authors], errors="coerce"))
    # age_at_travis_months = months between createdAt and adoption
    age_days = (df["ci_adoption_time"] - df[cfg.col_created_at]).dt.days
    df["age_at_travis_months"] = _days_to_months(age_days)

    # fill grouping
    if "bucket_id" not in df.columns:   df["bucket_id"]   = "overall"
    if "bucket_name" not in df.columns: df["bucket_name"] = "overall"

    return df.dropna(subset=["repo"]).reset_index(drop=True)

def build_panel_for_group(group_index: pd.DataFrame, cfg: RDDConfig) -> pd.DataFrame:
    """
    Build pooled monthly panel for a group (overall or bucket).
    Adds per-repo constant controls to each row.
    """
    rows = []
    meta_cols = ["ci_adoption_time", "mainLanguage", "log_total_commits",
                 "log_num_authors", "age_at_travis_months"]
    meta = group_index.set_index("repo")[meta_cols].to_dict(orient="index")

    for repo in group_index["repo"].unique():
        info = meta.get(repo)
        if info is None or pd.isna(info["ci_adoption_time"]):
            continue

        m = build_monthly_metrics_for_repo(repo)
        if m is None or m.empty:
            continue

        m["repo"] = repo
        m["mainLanguage"] = group_index.loc[group_index["repo"]==repo, "mainLanguage"].iloc[0]

        m = _build_design(m, info["ci_adoption_time"], skip_transition_month=cfg.skip_transition_month)
        if m.empty:
            continue

        # attach controls (constant within repo)
        m["log_total_commits"]    = info["log_total_commits"]
        m["log_num_authors"]      = info["log_num_authors"]
        m["age_at_travis_months"] = info["age_at_travis_months"]

        rows.append(m)

    if not rows:
        return pd.DataFrame(columns=["repo", "mainLanguage", "month"] + METRICS +
                                   ["time","intervention","time_after",
                                    "log_total_commits","log_num_authors","age_at_travis_months"])
    return pd.concat(rows, ignore_index=True)

# ------------------ Batch analysis & save ------------------

def analyze_dataset(index_df: pd.DataFrame,
                    groupby_bucket: bool,
                    cfg: RDDConfig,
                    source_tag: str,
                    out_path: Path) -> pd.DataFrame:
    """Run RDD for overall or per-bucket groups across all METRICS; save CSV."""
    results = []
    groups = [(("overall","overall"), index_df)]
    if groupby_bucket and "bucket_id" in index_df.columns:
        groups = list(index_df.groupby(["bucket_id","bucket_name"]))

    for (gid, gname), gdf in tqdm(groups, desc=f"{source_tag} groups"):
        panel = build_panel_for_group(gdf, cfg)
        if panel.empty:
            continue

        for metric in METRICS:
            stats = fit_rdd(panel, metric, cfg)

            # direction by delta
            b_d, p_d = stats.get("coef_slope_delta"), stats.get("p_slope_delta")
            if b_d is None or pd.isna(b_d) or p_d is None or pd.isna(p_d):
                direction = "NA"
            elif p_d <= 0.05:
                direction = "up" if b_d > 0 else "down" if b_d < 0 else "flat"
            else:
                direction = "ns"

            row = {
                "source": source_tag,
                "group_id": gid,
                "group_name": gname,
                "metric": metric,
                "mixed_effects": cfg.mixed_effects,
                "repo_RE": cfg.repo_random_intercept,
                "repo_RE_slope_intervention": cfg.repo_random_slope_intervention,
                "lang_RE": cfg.language_random_intercept,
                "lang_FE": cfg.language_fixed_effects,
                "controls": cfg.add_controls,
                "skip_transition": cfg.skip_transition_month,
                "robust_cov": cfg.robust_cov,
                "apply_log1p_y": cfg.apply_log1p_y,
                "trim_outliers_1pct": cfg.trim_outliers_1pct,
                "n_obs": stats.get("n_obs", 0),
                "n_projects": stats.get("n_projects", 0),

                # core coefficients
                "coef_time": stats.get("coef_time"),
                "se_time": stats.get("se_time"),
                "p_time": stats.get("p_time"),
                "stars_time": stats.get("stars_time"),

                "coef_level_gamma": stats.get("coef_level_gamma"),
                "se_level_gamma": stats.get("se_level_gamma"),
                "p_level_gamma": stats.get("p_level_gamma"),
                "stars_level_gamma": stats.get("stars_level_gamma"),

                "coef_slope_delta": stats.get("coef_slope_delta"),
                "se_slope_delta": stats.get("se_slope_delta"),
                "p_slope_delta": stats.get("p_slope_delta"),
                "stars_slope_delta": stats.get("stars_slope_delta"),

                # controls
                "coef_log_total_commits": stats.get("coef_log_total_commits"),
                "se_log_total_commits": stats.get("se_log_total_commits"),
                "p_log_total_commits": stats.get("p_log_total_commits"),
                "stars_log_total_commits": stats.get("stars_log_total_commits"),

                "coef_age_at_travis_months": stats.get("coef_age_at_travis_months"),
                "se_age_at_travis_months": stats.get("se_age_at_travis_months"),
                "p_age_at_travis_months": stats.get("p_age_at_travis_months"),
                "stars_age_at_travis_months": stats.get("stars_age_at_travis_months"),

                "coef_log_num_authors": stats.get("coef_log_num_authors"),
                "se_log_num_authors": stats.get("se_log_num_authors"),
                "p_log_num_authors": stats.get("p_log_num_authors"),
                "stars_log_num_authors": stats.get("stars_log_num_authors"),

                # ANOVA-like (Type-II Wald)
                "wald_time": stats.get("wald_time"),
                "wald_intervention": stats.get("wald_intervention"),
                "wald_time_after": stats.get("wald_time_after"),

                # R2
                "r2_marginal": stats.get("r2_marginal"),
                "r2_conditional": stats.get("r2_conditional"),

                # VIF
                "vif_max": stats.get("vif_max"),
                "vif_terms": stats.get("vif_terms"),

                "direction_by_delta": direction,
            }
            results.append(row)

    cols = [
        "source","group_id","group_name","metric",
        "mixed_effects","repo_RE","repo_RE_slope_intervention","lang_RE","lang_FE",
        "controls","skip_transition","robust_cov","apply_log1p_y","trim_outliers_1pct",
        "n_obs","n_projects",
        "coef_time","se_time","p_time","stars_time",
        "coef_level_gamma","se_level_gamma","p_level_gamma","stars_level_gamma",
        "coef_slope_delta","se_slope_delta","p_slope_delta","stars_slope_delta",
        "coef_log_total_commits","se_log_total_commits","p_log_total_commits","stars_log_total_commits",
        "coef_age_at_travis_months","se_age_at_travis_months","p_age_at_travis_months","stars_age_at_travis_months",
        "coef_log_num_authors","se_log_num_authors","p_log_num_authors","stars_log_num_authors",
        "wald_time","wald_intervention","wald_time_after",
        "r2_marginal","r2_conditional",
        "vif_max","vif_terms",
        "direction_by_delta",
    ]
    out = pd.DataFrame(results, columns=cols) if results else pd.DataFrame(columns=cols)
    out = out.sort_values(["source","group_id","metric"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


# ------------------ Runner (in-code config, no CLI) ------------------

@dataclass
class PipelineParams:
    input_source: str = "stra"         # "stra" -> base.csv, "quota" -> quota.csv
    groupby: str = "bucket"            # "bucket" or "overall"
    # modeling toggles
    mixed_effects: bool = True
    repo_random_intercept: bool = True            # kept for clarity; MixedLM uses it
    language_random_intercept: bool = True        # vc for language
    language_fixed_effects: bool = False          # paper uses RE; keep False
    add_controls: bool = True                     # add size/age/authors
    skip_transition_month: bool = True
    min_obs: int = 8
    robust_cov: str = "HC1"                       # OLS only
    # column mapping (replace when using different names)
    col_total_commits: str = "commits"
    col_num_authors: str  = "contributors"
    col_created_at: str   = "createdAt"
    # output
    out_path: Optional[str] = None

def run_rq3_pipeline(params: PipelineParams) -> pd.DataFrame:
    """Run full pipeline with in-code params."""
    idx_path   = STRA_CSV if params.input_source == "stra" else QUOTA_CSV
    source_tag = "STRA" if params.input_source == "stra" else "QUOTA"

    cfg_local = RDDConfig(
        mixed_effects=params.mixed_effects,
        repo_random_intercept=params.repo_random_intercept,
        language_random_intercept=params.language_random_intercept,
        language_fixed_effects=params.language_fixed_effects,
        skip_transition_month=params.skip_transition_month,  # << correct field
        min_obs=params.min_obs,
        robust_cov=params.robust_cov,
        max_iter=300,
        add_controls=params.add_controls,
        col_total_commits=params.col_total_commits,
        col_num_authors=params.col_num_authors,
        col_created_at=params.col_created_at,
    )

    # Load index and build+fit per group
    index_df = load_index(idx_path, cfg_local)

    # Output path
    out_path = (
        Path(params.out_path)
        if params.out_path
        else (DEFAULT_OUT_BUCKET if params.groupby == "bucket" else DEFAULT_OUT_OVERALL)
    )

    groupby_bucket = (params.groupby == "bucket")
    out = analyze_dataset(index_df, groupby_bucket, cfg_local, source_tag, out_path)

    # Preview
    with pd.option_context("display.max_rows", 50, "display.max_columns", 50, "display.width", 140):
        print(out.head(30))
        print(f"\nSaved results to: {out_path}")
    return out

# -------- Batch run: STRA overall + QUOTA overall + STRA by bucket --------
if __name__ == "__main__":
    # Global modeling defaults (paper-aligned)
    base_kwargs = dict(
        mixed_effects=True,               # Use MixedLM (paper setting)
        repo_random_intercept=True,       # (1 | repo)
        language_random_intercept=True,   # (1 | language) via vc
        language_fixed_effects=False,     # Paper models language as RE, not FE
        add_controls=True,                # log_total_commits / age_at_travis / log_num_authors
        skip_transition_month=True,       # drop the adoption month
        min_obs=8,                        # min observations per fitted panel
        robust_cov="HC1",                 # OLS-only; ignored by MixedLM
        col_total_commits="commits",
        col_num_authors="contributors",
        col_created_at="createdAt",
    )

    # You can override per run if needed (kept same here for comparability)
    stra_overall_kwargs = dict(base_kwargs)
    quota_overall_kwargs = dict(base_kwargs)
    stra_bucket_kwargs   = dict(base_kwargs)

    # 1) STRA overall (single pooled dataset)
    params_stra_overall = PipelineParams(
        input_source="stra",
        groupby="overall",
        out_path=str(RQ3_DATA_PATH / "rdd_overall_stra.csv"),
        **stra_overall_kwargs
    )
    print("\n=== Running STRA overall ===")
    run_rq3_pipeline(params_stra_overall)

    # 2) QUOTA overall (single pooled dataset)
    params_quota_overall = PipelineParams(
        input_source="quota",
        groupby="overall",
        out_path=str(RQ3_DATA_PATH / "rdd_overall_quota.csv"),
        **quota_overall_kwargs
    )
    print("\n=== Running QUOTA overall ===")
    run_rq3_pipeline(params_quota_overall)

    # 3) STRA by bucket (each bucket_id is a subdataset)
    params_stra_bucket = PipelineParams(
        input_source="stra",
        groupby="bucket",
        out_path=str(RQ3_DATA_PATH / "rdd_by_bucket_stra.csv"),
        **stra_bucket_kwargs
    )
    print("\n=== Running STRA by bucket ===")
    run_rq3_pipeline(params_stra_bucket)

    print("\nAll three runs finished.")