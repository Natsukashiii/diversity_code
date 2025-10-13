import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from config.load_config import load_config

# ----------------------------- Report helpers -----------------------------
cfg = load_config()
RQ1_DATA_PATH = Path(str(cfg.path.rq1_path))
RQ2_DATA_PATH = Path(str(cfg.path.rq2_path))
DATASET_DIR = Path(str(cfg.path.dataset_path))
DATASET_INFO_DIR = Path(str(cfg.path.dataset_info_path))
BASE_CSV = RQ1_DATA_PATH / "base.csv"
ANALYSIS_CSV = RQ1_DATA_PATH / "quota.csv"


def summarize_csv(
    data: Union[pd.DataFrame, str, Path],
    fields: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Summarize metadata fields and return as a compact table.
    
    - Numeric fields: min, max, mean
    - Datetime fields: min, max
    - Categorical fields: top categories + counts (moved to bottom of table)
    """
    default_fields = [
        "releases", "commits", "watchers", "stargazers",
        "contributors", "totalIssues", "openIssues",
        "mainLanguage", "totalPullRequests", "ci_type"
    ]
    fields = fields or default_fields

    # Handle input type
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Unsupported type for 'data': {type(data)}")

    numeric_summary, categorical_summary = [], []

    for col in fields:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_summary.append({
                "field": col,
                "type": "numeric",
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "extra": ""
            })

        elif pd.api.types.is_datetime64_any_dtype(series):
            numeric_summary.append({
                "field": col,
                "type": "datetime",
                "min": str(series.min()),
                "max": str(series.max()),
                "mean": "",
                "extra": ""
            })

        else:  # categorical
            counts = series.value_counts().head(5)
            extra = "; ".join(f"{val}:{cnt}" for val, cnt in counts.items())
            categorical_summary.append({
                "field": col,
                "type": "categorical",
                "min": "",
                "max": "",
                "mean": "",
                "extra": extra
            })

    # concat: numeric first, categorical last
    summary_df = pd.DataFrame(numeric_summary + categorical_summary)

    print("\n===== Data Summary (Table) =====")
    print(summary_df.to_string(index=False))
    print("===== End of Summary =====\n")

    return summary_df



# ----------------------------- I/O helpers -----------------------------
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV; raise if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata not found: {p}")
    return pd.read_csv(p)


def safe_to_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV with parent creation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

    
def load_source(path, source_type="csv"):
    if source_type == "csv":
        return load_csv(path)
    elif source_type == "directory":
        return load_from_directory(path)
    else:

        raise ValueError(f"Unknown source type: {source_type}")

def load_from_directory(directory_path):
    all_files = []
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            df = load_csv(os.path.join(directory_path, file))
            all_files.append(df)
    return pd.concat(all_files, ignore_index=True)






def load_repo_meta_from_csv(base_csv: Path) -> pd.DataFrame:
    """Load base.csv minimal columns."""
    if not base_csv.exists():
        return pd.DataFrame(columns=["repo", "defaultBranch", "mainLanguage", "ci_type", "ci_adoption_time","bucket_id", "bucket_name"])
    df = pd.read_csv(base_csv, low_memory=False)
    df["repo"] = df["repo"].astype(str).str.strip()
    keep = [c for c in ["repo", "defaultBranch", "mainLanguage", "ci_type", "ci_adoption_time","bucket_id","bucket_name"] if c in df.columns]
    return df[keep].drop_duplicates(subset=["repo"], keep="first")


def _load_repo_meta(source_mode: str) -> pd.DataFrame:
    """Load minimal repo meta with bucket columns."""
    if source_mode.lower() == "dataset":
        df = scan_repos_from_dataset(DATASET_DIR, BASE_CSV)
    else:
        df = load_repo_meta_from_csv(ANALYSIS_CSV)

    keep = [c for c in ["repo","defaultBranch","mainLanguage","ci_type","ci_adoption_time","bucket_id","bucket_name"] if c in df.columns]
    out = df[keep].drop_duplicates(subset=["repo"], keep="first").copy()
    out["bucket_name"] = out.get("bucket_name", pd.Series(dtype="object")).apply(_clean_code)
    if "bucket_id" in out.columns:
        out = out.dropna(subset=["bucket_id"])
    return out

# _CODE_RE = re.compile(r"^[PJ][GT][Aa][Ii]$")
def _clean_code(x) -> Optional[str]:
    return str(x).strip() if pd.notna(x) else None
    # s = str(x).strip() if pd.notna(x) else ""
    # return s if _CODE_RE.match(s) else None


def scan_repos_from_dataset(dataset_dir: Path, base_csv: Path) -> pd.DataFrame:
    """Scan folders in dataset_dir and LEFT JOIN base.csv."""
    repos: List[str] = []
    if dataset_dir.exists():
        for owner in dataset_dir.iterdir():
            if not owner.is_dir():
                continue
            for repo in owner.iterdir():
                if repo.is_dir():
                    repos.append(f"{owner.name}/{repo.name}")
    df_repos = pd.DataFrame({"repo": repos})
    meta = load_repo_meta_from_csv(base_csv)
    return df_repos.merge(meta, on="repo", how="left")



def repo_dirs(dataset_info_dir: Path, full_name: str) -> Path:
    """dataset_info_dir/owner/repo"""
    owner, name = full_name.split("/", 1)
    return dataset_info_dir / owner / name



def summarize_repo_counts_by_bucket(ts_csv: Path, out_csv: Path) -> None:
    """
    calculate the repos
    :bucket_id, bucket_name, repo_count
    """
    usecols = ["repo", "bucket_id", "bucket_name"]
    df = _read_csv_lazy(ts_csv, usecols=usecols)
    if df.empty:
        out_csv.write_text("")  # touch an empty file
        return

    df["bucket_id"] = pd.to_numeric(df["bucket_id"], errors="coerce")
    grp = (
        df.dropna(subset=["bucket_id", "repo"])
          .groupby(["bucket_id", "bucket_name"], dropna=False)["repo"]
          .nunique()
          .reset_index(name="repo_count")
          .sort_values(["bucket_id", "bucket_name"], kind="mergesort")
    )
    grp.to_csv(out_csv, index=False)