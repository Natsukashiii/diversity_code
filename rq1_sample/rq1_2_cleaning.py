# 3_data_cleaning.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.load_config import load_config

# ---------------------------- config ----------------------------
cfg = load_config()
RQ1_DATA_PATH = Path(str(cfg.path.rq1_path))
DATASET_INFO_DIR = Path(str(cfg.path.dataset_info_path))

NEED_CLEAN_CSV = RQ1_DATA_PATH / "metadata.csv"
OUTPUT_CSV = RQ1_DATA_PATH / "metadata_cleaned.csv"

# ---------------------------- logging ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------- helpers ----------------------------

LIKELY_DT_COL_REGEX = re.compile(
    r"(time|date|created|updated|published|started|finished|"
    r"pushed|committed|merged|closed|opened|timestamp)",
    re.IGNORECASE,
)

COMMON_DT_FORMATS = [
    "ISO8601",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]

def _normalize_repo(full_name: str) -> str:
    """Normalize 'owner/repo' (trim, lowercase); invalid -> ''."""
    if not isinstance(full_name, str):
        return ""
    s = full_name.strip().strip("/").lower()
    return s if "/" in s else ""

def _repo_info_dir(repo_full: str) -> Path:
    """DATASET_INFO_DIR/owner/repo."""
    owner, name = repo_full.split("/", 1)
    return DATASET_INFO_DIR / owner / name

def _csv_has_rows(path: Path) -> bool:
    """Return True if CSV exists and has >=1 data row."""
    if not path.exists() or not path.is_file():
        return False
    try:
        df = pd.read_csv(path, nrows=1, low_memory=False)
        return df.shape[0] > 0
    except Exception:
        return False

def _check_required_files_nonempty(repo_full: str) -> Dict[str, bool]:
    """Check required CSV availability for a repo."""
    root = _repo_info_dir(repo_full)
    files = {
        "commits_local": root / "commits_local.csv",
        "prs":           root / "prs.csv",
        "releases":      root / "releases.csv",
        "issues":        root / "issues.csv",
        "gactions":      root / "gactions.csv",
        "jobs":          root / "jobs.csv",
    }
    return {k: _csv_has_rows(p) for k, p in files.items()}

def _parse_ts_series(s: pd.Series) -> pd.Series:
    """Parse timestamps to UTC; invalid -> NaT."""
    return pd.to_datetime(s, errors="coerce", utc=True)

def _earliest_ts_in_csv(path: Path) -> Optional[pd.Timestamp]:
    """
    Load a CSV (low_memory=False) and return the earliest timestamp found across
    ALL columns after attempting to parse them as datetimes.
    Returns None if file missing or no parseable timestamps.
    """
    if not path.exists() or not path.is_file():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None

    cand_cols = [c for c in df.columns if LIKELY_DT_COL_REGEX.search(str(c))]
    if not cand_cols:
        return None

    earliest: Optional[pd.Timestamp] = None

    for col in cand_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue

        s = s.astype("string", errors="ignore")

        parsed = None
        try:
            parsed = pd.to_datetime(s, format="ISO8601", errors="coerce", utc=True)
        except Exception:
            parsed = None

        if parsed is None or parsed.isna().all():
            for fmt in COMMON_DT_FORMATS:
                try:
                    parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
                except Exception:
                    parsed = None
                if parsed is not None and not parsed.isna().all():
                    break

        if parsed is None or parsed.isna().all():
            continue

        v = parsed.min(skipna=True)
        if pd.isna(v):
            continue

        if earliest is None or v < earliest:
            earliest = v

    return earliest

def _log_time_diagnostics(df: pd.DataFrame,
                          label: str,
                          min_days_created_to_adopt: float,
                          min_days_adopt_to_now: float) -> None:
    """Log diagnostics for created/adoption times and deltas."""
    n = len(df)
    if n == 0:
        logging.info(f"[diag:{label}] empty frame")
        return

    c_nat = int(df["createdAt"].isna().sum()) if "createdAt" in df.columns else -1
    a_nat = int(df["ci_adoption_time"].isna().sum()) if "ci_adoption_time" in df.columns else -1
    logging.info(f"[diag:{label}] rows={n}, createdAt_NaT={c_nat}, adoption_NaT={a_nat}")

    if "days_created_to_adopt" in df.columns and "days_adopt_to_now" in df.columns:
        d1 = df["days_created_to_adopt"]
        d2 = df["days_adopt_to_now"]
        fail_d1 = int((d1 <= min_days_created_to_adopt).sum())
        fail_d2 = int((d2 <= min_days_adopt_to_now).sum())
        q = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        q1 = d1.dropna().quantile(q).to_dict() if d1.notna().any() else {}
        q2 = d2.dropna().quantile(q).to_dict() if d2.notna().any() else {}
        logging.info(
            f"[diag:{label}] created→adopt: fail<={min_days_created_to_adopt:.1f}d={fail_d1}, "
            f"quantiles={{{' | '.join(f'{k:.2f}:{v:.1f}' for k,v in q1.items())}}}"
        )
        logging.info(
            f"[diag:{label}] adopt→now:     fail<={min_days_adopt_to_now:.1f}d={fail_d2}, "
            f"quantiles={{{' | '.join(f'{k:.2f}:{v:.1f}' for k,v in q2.items())}}}"
        )



def _determine_ci_by_files(repo_full: str) -> Tuple[str, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Inspect dataset_info/<owner>/<repo> to detect CI evidence and earliest timestamps.

    Returns
    -------
    ci_hint : str
        "TravisCI" if jobs.csv exists and has rows, else "GitHubActions" if gactions.csv has rows,
        otherwise "unknown".
    jobs_ts : Optional[pd.Timestamp]
        Earliest timestamp parsed from jobs.csv (UTC), or None.
    gact_ts : Optional[pd.Timestamp]
        Earliest timestamp parsed from gactions.csv (UTC), or None.
    """
    root = _repo_info_dir(repo_full)
    jobs_csv = root / "jobs.csv"
    gact_csv = root / "gactions.csv"

    has_jobs = _csv_has_rows(jobs_csv)
    has_gact = _csv_has_rows(gact_csv)

    jobs_ts = _earliest_ts_in_csv(jobs_csv) if has_jobs else None
    gact_ts = _earliest_ts_in_csv(gact_csv) if has_gact else None

    if has_jobs:
        ci_hint = "TravisCI"
    elif has_gact:
        ci_hint = "GitHubActions"
    else:
        ci_hint = "unknown"

    # Debug trace to help diagnose missing timestamps
    logging.debug(
        "[ci-scan] %s -> hint=%s, has_jobs=%s, has_gact=%s, jobs_ts=%s, gact_ts=%s",
        repo_full, ci_hint, has_jobs, has_gact, jobs_ts, gact_ts
    )
    return ci_hint, jobs_ts, gact_ts


def _choose_ci_type_and_adoption_backup(repo_full: str,
                                 orig_ci_type: Optional[str],
                                 created_at: Optional[pd.Timestamp]
                                 ) -> Tuple[Optional[str], Optional[pd.Timestamp], bool]:
    """
    Decide final (ci_type, ci_adoption_time, drop_flag) by rules:

    1) If jobs.csv has data → force ci_type='TravisCI', adopt=earliest jobs.csv ts.
    2) Else keep original ci_type (default to 'GitHubActions' if missing).
       If ci_type=='GitHubActions' → adopt=earliest gactions.csv ts (may be None).
    3) If adopt < createdAt → try fallback to gactions.csv ts; if still None → drop_flag=True.
    4) NEW fallback for adopt is None:
       - If ci_type=='TravisCI' and jobs_ts is None but gact_ts exists → switch to GitHubActions and use gact_ts.
       - If ci_type=='GitHubActions' and gact_ts is None but jobs_ts exists → switch to TravisCI and use jobs_ts.
       - If both None → keep adopt=None (caller/Rule1 will drop), drop_flag=False here.

    Returns
    -------
    ci_type : Optional[str]
    adopt   : Optional[pd.Timestamp]
    drop    : bool    # True when we explicitly decide to drop the row due to inconsistency.
    """
    ci_files_type, jobs_ts, gact_ts = _determine_ci_by_files(repo_full)
    orig_ci = (orig_ci_type or "").strip()

    # Rule 1: jobs.csv present → TravisCI with earliest jobs timestamp
    if ci_files_type == "TravisCI" and jobs_ts is not None:
        ci_type = "TravisCI"
        adopt = jobs_ts
    else:
        # Rule 2: fallback to original ci_type (prefer GA if unknown)
        ci_type = orig_ci if orig_ci else "GitHubActions"
        adopt = gact_ts if ci_type == "GitHubActions" else None

    # print(f"[debug]Rules1-2: repo={repo_full}, orig_ci={orig_ci}, ci_files_type={ci_files_type}, jobs_ts={jobs_ts}, gact_ts={gact_ts} => ci_type={ci_type}, adopt={adopt}")

    # Rule 3: adoption earlier than creation → try gactions fallback, else drop
    drop_row = False
    if adopt is not None and created_at is not None and adopt < created_at:
        if gact_ts is not None:
            ci_type = "GitHubActions"
            adopt = gact_ts
            logging.debug("[fix] %s adoption<created; switched to GA adopt=%s", repo_full, adopt)
        else:
            logging.warning(
                "[warn] %s: adoption %s < created %s and no gactions fallback → drop",
                repo_full, adopt, created_at
            )
            drop_row = True
    
    # print(f"[debug]Rule3: repo={repo_full}, created_at={created_at} => ci_type={ci_type}, adopt={adopt}, drop_row={drop_row}")

    # Rule 4: adopt is still None → cross-fallback using the *other* CI file
    if adopt is None and not drop_row:
        if ci_type == "TravisCI" and gact_ts is not None:
            # Prefer to fix missing Travis time with Actions evidence
            ci_type = "GitHubActions"
            adopt = gact_ts
            logging.debug("[fallback] %s no jobs ts; switched to GA adopt=%s", repo_full, adopt)
        elif ci_type == "GitHubActions" and jobs_ts is not None:
            # Or fix missing Actions time with Travis evidence
            ci_type = "TravisCI"
            adopt = jobs_ts
            logging.debug("[fallback] %s no gactions ts; switched to Travis adopt=%s", repo_full, adopt)

    # Final debug line for this repo
    return ci_type, adopt, drop_row



# ---------------------------- core cleaning ----------------------------
def clean_metadata(
    in_csv: Path,
    out_csv: Path,
    *,
    min_years_since_created_to_adopt: float = 1.0,
    min_years_since_adopt_to_now: float = 1.0,
) -> pd.DataFrame:
    """Apply cleaning rules and save to out_csv; print per-step drop counts."""
    if not in_csv.exists():
        raise FileNotFoundError(f"metadata file not found: {in_csv}")

    # Load raw
    df = pd.read_csv(in_csv, low_memory=False)
    if "repo" not in df.columns:
        raise ValueError("metadata.csv must contain 'repo' column")

    total_raw = len(df)

    # Normalize repo and drop invalid
    df["repo"] = df["repo"].astype(str).map(_normalize_repo)
    before = len(df)
    df = df[df["repo"] != ""].copy()
    drop_invalid_repo = before - len(df)

    # Parse createdAt (from metadata)
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True, format="ISO8601")

    # --- Use original columns only; no inference/fallback ---
    # Ensure ci_type exists (keep as-is from original CSV)
    if "ci_type" not in df.columns:
        df["ci_type"] = None

    # Normalize original ci_adoption_time to UTC/NaT
    df["ci_adoption_time"] = pd.to_datetime(df.get("ci_adoption_time"), errors="coerce", utc=True)

    # Drop rows where original adoption is earlier than repository creation
    before = len(df)
    bad_mask = (
        df["ci_adoption_time"].notna()
        & df["createdAt"].notna()
        & (df["ci_adoption_time"] < df["createdAt"])
    )
    df = df[~bad_mask].copy()
    drop_bad_time = before - len(df)


    # Rule 1: drop rows without ci_adoption_time
    before = len(df)
    df = df[~df["ci_adoption_time"].isna()].copy()
    drop_rule1 = before - len(df)

    # Rule 2: time windows (> 1 year from created→adopt AND > 1 year from adopt→now)
    now = pd.Timestamp.now(tz="UTC")
    year_days = 365
    min_days_created_to_adopt = min_years_since_created_to_adopt * year_days
    min_days_adopt_to_now = min_years_since_adopt_to_now * year_days

    df["days_created_to_adopt"] = (df["ci_adoption_time"] - df["createdAt"]).dt.total_seconds() / 86400.0
    df["days_adopt_to_now"] = (now - df["ci_adoption_time"]).dt.total_seconds() / 86400.0

    # diagnostics BEFORE filtering Rule2
    _log_time_diagnostics(
        df,
        label="before_rule2",
        min_days_created_to_adopt=min_days_created_to_adopt,
        min_days_adopt_to_now=min_days_adopt_to_now,
    )

    before = len(df)
    df = df[
        (df["days_created_to_adopt"] > min_days_created_to_adopt) &
        (df["days_adopt_to_now"] > min_days_adopt_to_now)
    ].copy()
    drop_rule2 = before - len(df)
    logging.info(
    f"[rule2] strictly >{min_years_since_created_to_adopt}y (created→adopt) "
    f"AND >{min_years_since_adopt_to_now}y (adopt→now) dropped: {drop_rule2} "
)

    # diagnostics AFTER filtering Rule2
    _log_time_diagnostics(
        df,
        label="after_rule2",
        min_days_created_to_adopt=min_days_created_to_adopt,
        min_days_adopt_to_now=min_days_adopt_to_now,
    )

    # Rule 3: check required files non-empty
    # 3.1 any of commits_local/prs/releases/issues empty -> drop
    # 3.2 gactions and jobs both empty -> drop
    keep_mask = []
    core_empty_flags = []
    ci_both_empty_flags = []

    for r in df.itertuples(index=False):
        repo_full = getattr(r, "repo")
        checks = _check_required_files_nonempty(repo_full)

        core_empty = any(not checks[k] for k in ["commits_local", "prs", "releases", "issues"])
        both_ci_empty = (not checks["gactions"]) and (not checks["jobs"])

        core_empty_flags.append(core_empty)
        ci_both_empty_flags.append(both_ci_empty)

        keep_mask.append((not core_empty) and (not both_ci_empty))

    before = len(df)
    mask_keep = pd.Series(keep_mask, index=df.index)
    mask_drop = ~mask_keep

    # actual dropped by Rule3
    df = df.loc[mask_keep].copy()
    drop_rule3 = before - len(df)

    # only in dropped samples count specific reasons (note that both reasons may be true at the same time)
    core_empty_count = int((mask_drop & pd.Series(core_empty_flags, index=mask_drop.index)).sum())
    ci_both_empty_count = int((mask_drop & pd.Series(ci_both_empty_flags, index=mask_drop.index)).sum())

    # Save cleaned CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in ["days_created_to_adopt", "days_adopt_to_now"] if c in df.columns], inplace=True)
    df.to_csv(out_csv, index=False)

    # ---------------------------- summary prints ----------------------------
    logging.info("========== Cleaning Summary ==========")
    logging.info(f"Loaded raw rows                : {total_raw}")
    logging.info(f"Drop invalid repo format       : {drop_invalid_repo}")
    logging.info(f"Drop bad CI time (pre-Rule1)   : {drop_bad_time}     ")
    logging.info(f"Rule1 no ci_adoption_time      : {drop_rule1}")
    logging.info(f"Rule2 time-window (>1y)        : {drop_rule2}")
    logging.info(f"Rule3 dropped by file checks   : {drop_rule3}")
    logging.info(f"  └─ of which core file empty  : {core_empty_count}  (commits/prs/releases/issues)")
    logging.info(f"  └─ of which CI logs both 0   : {ci_both_empty_count}  (gactions & jobs)")
    logging.info(f"Final kept rows                : {len(df)}")
    logging.info(f"Saved                          : {out_csv}")

    print("---- CLEANING SUMMARY ----")
    print(
        f"raw={total_raw}, drop_invalid_repo={drop_invalid_repo}, "
        f"drop_bad_time={drop_bad_time}, drop_rule1={drop_rule1}, "
        f"drop_rule2={drop_rule2}, drop_rule3={drop_rule3}, "
        f"final={len(df)}"
    )
    return df

# ---------------------------- main ----------------------------
if __name__ == "__main__":
    print(f"INPUT: {NEED_CLEAN_CSV}")
    cleaned = clean_metadata(
        NEED_CLEAN_CSV,
        OUTPUT_CSV,
        min_years_since_created_to_adopt=1.0,
        min_years_since_adopt_to_now=1.0,
    )
    print(f"FINAL REPOS: {len(cleaned)}  →  {OUTPUT_CSV}")