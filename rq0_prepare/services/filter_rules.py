from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from services.filter_helper import (CIAssignment, normalize_lang,
                                    parse_ci_adoption_times, parse_iso,
                                    pick_ci, to_bool, years_between)

# Required input columns (validated before filtering)
REQUIRED_COLUMNS: List[str] = [
    "repo", "isFork", "isArchived", "isDisabled", "isLocked",
    "defaultBranch", "releases", "commits", "watchers", "stargazers",
    "contributors", "createdAt", "totalIssues", "openIssues",
    "mainLanguage", "lastCommit", "totalPullRequests", "ci_adoption_times"
]

# Output columns (after filtering/derivation)
OUTPUT_COLUMNS: List[str] = [
    "repo",
    "defaultBranch", "size", "releases", "commits", "watchers", "stargazers",
    "contributors", "createdAt", "totalIssues", "openIssues",
    "mainLanguage", "lastCommit", "totalPullRequests",
    "ci_type", "ci_adoption_time",
    "avg_contribution",
    "open_issue_ratio",
    "issues_per_day",         
    "commits_per_day",  
    "project_age_days"  ,
    "avg_issues_close_time",
    "test_percentage",  
    "size_per_dev", 
]


# ---------- individual rules ----------

def rule_repo_state(row: pd.Series) -> bool:
    """Require isArchived=False, isDisabled=False, isLocked=False."""
    for col in ("isArchived", "isDisabled", "isLocked"):
        v = to_bool(row.get(col))
        if v is None or v:  # None -> reject to be strict
            return False
    return True


def rule_releases_positive(row: pd.Series) -> bool:
    """Require releases > 0."""
    try:
        return float(row.get("releases", 0)) > 12
    except Exception:
        return False


def rule_commits_density(row: pd.Series, now: Optional[datetime] = None) -> bool:
    """Require commits / age(years) > 12, age = now - createdAt."""
    now = now or datetime.now(timezone.utc)
    created = parse_iso(row.get("createdAt"))
    if not created:
        return False
    try:
        commits = float(row.get("commits", 0))
    except Exception:
        return False
    age_years = max(years_between(created, now), 1e-9)
    return (commits / age_years) > 12.0


def rule_issues_positive(row: pd.Series) -> bool:
    """Require totalIssues > 0."""
    try:
        return float(row.get("totalIssues", 0)) > 12
    except Exception:
        return False


def rule_language(row: pd.Series) -> bool:
    """Require mainLanguage in {'java','python'} (case-insensitive)."""
    return normalize_lang(row.get("mainLanguage")) in {"java", "python"}


def rule_last_commit_in_2024(row: pd.Series) -> bool:
    """Require lastCommit >= 2024-01-01 UTC."""
    dt = parse_iso(row.get("lastCommit"))
    if dt is None:
        return False
    threshold = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return dt >= threshold


def rule_ci_adopted(row: pd.Series) -> CIAssignment:
    """Resolve CI choice/time from ci_adoption_times field."""
    ci_map = parse_ci_adoption_times(row.get("ci_adoption_times"))
    return pick_ci(ci_map)

# ---------- add fields helpers ----------
def enrich_metrics(df: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    """
    Add derived activity metrics:
      - commits_per_contributor: commits / contributors
      - open_issue_ratio:       openIssues / totalIssues  (clamped to [0,1])
      - issues_per_day:         totalIssues / project_age_days
      - commits_per_day:        commits / project_age_days
    """
    out = df.copy()
    now = now or datetime.now(timezone.utc)

    # commits / contributors (per-member activity)
    def _commits_per_contributor(row: pd.Series) -> Optional[float]:
        try:
            commits = float(row.get("commits", 0))
            contribs = float(row.get("contributors", 0))
        except Exception:
            return None
        if contribs <= 0:
            return None
        return commits / contribs

    # open issues ratio (health of backlog)
    def _open_issue_ratio(row: pd.Series) -> Optional[float]:
        try:
            total = float(row.get("totalIssues", 0))
            open_ = float(row.get("openIssues", 0))
        except Exception:
            return None
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, open_ / total))

    # helper: project age (days, >=1)
    def _age_days(row: pd.Series) -> int:
        created = parse_iso(row.get("createdAt"))
        if created is None:
            return 1
        return max((now - created).days, 1)

    # total issues per day (issue activity rate)
    def _issues_per_day(row: pd.Series) -> Optional[float]:
        try:
            total = float(row.get("totalIssues", 0))
        except Exception:
            return None
        return total / _age_days(row)

    # commits per day (commit activity rate)
    def _commits_per_day(row: pd.Series) -> Optional[float]:
        try:
            commits = float(row.get("commits", 0))
        except Exception:
            return None
        return commits / _age_days(row)
    
    # repo size per developer
    def _size_per_dev(row: pd.Series) -> Optional[float]:
        try:
            size = float(row.get("size", 0))
            contribs = float(row.get("contributors", 0))
        except Exception:
            return None
        if contribs <= 0:
            return None
        return size / contribs

    out["avg_contribution"] = out.apply(_commits_per_contributor, axis=1)
    out["open_issue_ratio"]        = out.apply(_open_issue_ratio, axis=1)
    out["issues_per_day"]          = out.apply(_issues_per_day, axis=1)
    out["commits_per_day"]         = out.apply(_commits_per_day, axis=1)
    out["project_age_days"]       = out.apply(_age_days, axis=1)
    out["size_per_dev"]     = out.apply(_size_per_dev, axis=1)


    return out



def merge_old_data(
    base_df: pd.DataFrame,
    old_data_path: Optional[Union[str, Path]],
    *,
    on: str = "repo",
    keep_cols: Optional[Iterable[str]] = None,
    normalize_key: bool = True,
    allow_overwrite: bool = False,
    suffix_old: str = "_old",
    missing_like: Iterable[str] = ("", "NA", "N/A", "null", "None"),
    report: bool = True,
) -> pd.DataFrame:
    """Merge extra columns from old CSV into base_df with debug prints."""
    if old_data_path in (None, "", False):
        if report: print("[INFO] old_data_path is None → skip merge")
        return base_df

    old_path = Path(old_data_path)
    if not old_path.exists():
        if report: print(f"[INFO] old_data not found: {old_path} → skip merge")
        return base_df

    try:
        old_df = pd.read_csv(old_path, low_memory=False)
    except Exception as e:
        if report: print(f"[WARN] failed to read old_data: {old_path} ({e}) → skip merge")
        return base_df

    base = base_df.copy()
    old  = old_df.copy()

    # normalize column names
    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
        return df
    base = _norm_cols(base)
    old  = _norm_cols(old)

    print(f"[DEBUG] base_df cols: {list(base.columns)[:10]} ... total={len(base.columns)}")
    print(f"[DEBUG] old_df cols: {list(old.columns)[:10]} ... total={len(old.columns)}")

    if on not in base.columns or on not in old.columns:
        if report: print(f"[WARN] merge key '{on}' missing → skip merge")
        return base

    if normalize_key:
        base[on] = base[on].astype(str).str.strip().str.lower()
        old[on]  = old[on].astype(str).str.strip().str.lower()

    # choose columns
    if keep_cols is None:
        bring = [c for c in old.columns if c != on]
    else:
        bring = [c.strip() for c in keep_cols if c in old.columns and c != on]
    print(f"[DEBUG] bring columns: {bring}")

    if not bring:
        if report: print("[INFO] no valid columns to merge → skip merge")
        return base

    old_small = old[[on] + bring].drop_duplicates(subset=[on], keep="first")
    print(f"[DEBUG] old_small shape={old_small.shape}")

    merged = base.merge(old_small, on=on, how="left", suffixes=("", suffix_old))
    print(f"[DEBUG] merged shape={merged.shape}")

    # check a few rows
    print("[DEBUG] sample rows after merge:")
    print(merged[[on] + [c for c in merged.columns if c.endswith(suffix_old)]].head(5))

    stats = {}
    for col in bring:
        temp = col + suffix_old
        if temp not in merged.columns:
            stats[col] = 0
            continue

        if col in merged.columns:
            if allow_overwrite:
                mask = merged[temp].notna()
            else:
                mask = merged[col].isna() & merged[temp].notna()
            stats[col] = int(mask.sum())
            merged.loc[mask, col] = merged.loc[mask, temp]
            merged.drop(columns=[temp], inplace=True)
        else:
            merged.rename(columns={temp: col}, inplace=True)
            stats[col] = int(merged[col].notna().sum())

    if report:
        inter = len(set(base[on]) & set(old_small[on]))
        print(f"[OK] merged: {bring} (rows={len(merged)}, key_intersect={inter}, overwrite={allow_overwrite})")
        for c, n in stats.items():
            print(f"   - {c}: applied={n}")

        # extra debug: show how many non-nulls per column
        for c in bring:
            if c in merged.columns:
                print(f"[DEBUG] col={c} notna count={merged[c].notna().sum()}")

    return merged
# ---------- pipeline helpers ----------

def validate_columns(df: pd.DataFrame) -> None:
    """Ensure all required columns exist."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all rules and produce ci_type/ci_adoption_time columns."""
    df = df.copy()

    # Resolve CI info upfront (vectorized over column)
    ci_series = df["ci_adoption_times"].apply(parse_ci_adoption_times).apply(pick_ci)
    df["ci_type"] = ci_series.apply(lambda x: x.ci_type)
    df["ci_adoption_time"] = ci_series.apply(lambda x: x.ci_adoption_time)

    m_state = df.apply(rule_repo_state, axis=1)
    m_releases = df.apply(rule_releases_positive, axis=1)
    m_density = df.apply(rule_commits_density, axis=1)
    m_issues = df.apply(rule_issues_positive, axis=1)
    m_lang = df.apply(rule_language, axis=1)
    m_last = df.apply(rule_last_commit_in_2024, axis=1)
    m_ci = df["ci_type"].isin({"GitHubActions", "TravisCI"}) & df["ci_adoption_time"].notna()

    mask = m_state & m_releases & m_density & m_issues & m_lang & m_last & m_ci
    return df.loc[mask, :]


def tidy_output(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only output columns and normalize types/format (guard missing cols)."""
    # Keep only columns that actually exist
    keep_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    out = df.reindex(columns=keep_cols).copy()

    # Normalize booleans if the columns exist
    for col in ("isFork", "isArchived", "isDisabled", "isLocked"):
        if col in out.columns:
            out[col] = out[col].apply(lambda v: bool(v) if isinstance(v, bool) else to_bool(v))

    # Format datetime column to ISO string if present
    if "ci_adoption_time" in out.columns:
        out["ci_adoption_time"] = out["ci_adoption_time"].apply(
            lambda d: d.isoformat() if hasattr(d, "isoformat") else None
        )

    return out