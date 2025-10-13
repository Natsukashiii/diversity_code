# services/common_helper.py

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


def summarize_bucket_counts(
    datasets: Dict[str, pd.DataFrame],
    out_csv: Path,
    *,
    repo_col: str = "repo",
    bucket_cols: Tuple[str, str, str] = ("bucket_id", "bucket_desc", "bucket_name"),
) -> Path:
    """
    For each dataset (name -> DataFrame), count unique repos per bucket.
    Output CSV columns: dataset, bucket_id, bucket_desc, bucket_name, n_repos.

    Parameters
    ----------
    datasets : dict
        Mapping from dataset name to its cohort dataframe.
        Each df is expected to have columns: repo, bucket_id, bucket_desc, bucket_name.
    out_csv : Path
        Where to save the combined CSV.
    repo_col : str
        Column name for repository full_name (owner/repo).
    bucket_cols : (id, desc, name)
        Column names for bucket fields.

    Returns
    -------
    Path
        The path to the saved CSV.
    """
    id_col, desc_col, name_col = bucket_cols
    frames = []

    for ds_name, df in datasets.items():
        if df is None or df.empty:
            continue
        needed = {repo_col, id_col, desc_col, name_col}
        if not needed.issubset(df.columns):
            # Skip gracefully if columns missing
            continue

        tmp = df[[repo_col, id_col, desc_col, name_col]].copy()

        # Normalize dtypes (robust across mixed types)
        tmp[repo_col] = tmp[repo_col].astype(str)
        tmp[id_col] = tmp[id_col].astype(str)

        grp = (
            tmp.groupby([id_col, desc_col, name_col], dropna=False)[repo_col]
               .nunique()
               .reset_index(name="n_repos")
        )
        grp.insert(0, "dataset", ds_name)
        frames.append(grp)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["dataset", id_col, desc_col, name_col, "n_repos"]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return out_csv



