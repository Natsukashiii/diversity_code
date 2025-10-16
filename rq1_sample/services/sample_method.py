# sample_method.py
from __future__ import annotations

from itertools import count
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def random_sample(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    """Simple random sampling without replacement."""
    n = min(sample_size, len(df))
    return df.sample(n=n, replace=False, random_state=random_state)


def _bucket_counts(df: pd.DataFrame, bucket_cols: List[str]) -> pd.Series:
    """Count elements per unique bucket (multi-index → flat Series)."""
    return df.groupby(bucket_cols, dropna=False).size().sort_values(ascending=False)


def stratified_sample(
    df_bucketed: pd.DataFrame,
    bucket_cols: List[str],
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Proportional stratified sampling:
    - allocate samples to buckets by their population proportion
    - round and fix totals; sample within each bucket without replacement
    """
    counts = _bucket_counts(df_bucketed, bucket_cols)
    total = int(counts.sum())
    if total == 0:
        return df_bucketed.iloc[0:0].copy()

    # ideal allocation
    alloc = (counts / total * sample_size).round().astype(int)

    # adjust rounding to hit exact sample_size
    diff = sample_size - int(alloc.sum())
    if diff != 0:
        # add/subtract from the largest buckets first
        order = counts.index
        signs = np.sign(diff)
        for idx in order:
            if diff == 0:
                break
            alloc.loc[idx] = max(0, alloc.loc[idx] + signs)
            diff -= signs

    # draw
    out = []
    for bucket_key, k in alloc.items():
        if k <= 0:
            continue
        mask = (df_bucketed[bucket_cols] == pd.Series(bucket_key, index=bucket_cols)).all(axis=1)
        pool = df_bucketed.loc[mask]
        k = min(k, len(pool))
        if k > 0:
            out.append(pool.sample(n=k, replace=False, random_state=random_state))
    return pd.concat(out, ignore_index=True) if out else df_bucketed.iloc[0:0].copy()




def quota_sample(
    df_bucketed: pd.DataFrame,
    bucket_cols: List[str],
    total_size: int,
    random_state: int = 42,
    expected_index: Optional[pd.MultiIndex] = None,  # optional: full bucket space
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Equal-quota sampling with stable bucket_id/desc for empty buckets.

    - target per bucket = floor(total_size / B), remainder -> largest buckets
    - sample k = min(target, available) without replacement
    - gaps use 'sampled': shortfall = max(target - sampled, 0)
    - if expected_index is given, include missing buckets (available=0) in gaps
    - bucket_id / bucket_desc are assigned for *all* buckets, even when available=0
    """

    # ---------- counts over buckets (optionally include zero-available buckets) ----------
    counts = df_bucketed.groupby(bucket_cols, dropna=False).size()
    if expected_index is not None:
        counts = counts.reindex(expected_index, fill_value=0)

    if counts.empty:
        empty = df_bucketed.iloc[0:0].copy()
        gaps_empty = pd.DataFrame(columns=["bucket_id", "bucket_desc", "target", "available", "sampled", "shortfall"])
        return empty, gaps_empty

    # ---------- stable mapping: (bucket tuple) -> bucket_id ----------
    # 1) map existing combinations from df (if df already has bucket_id, reuse it)
    #    else enumerate deterministically.
    if "bucket_id" in df_bucketed.columns:
        base_map = (
            df_bucketed.drop_duplicates(subset=bucket_cols)
            .set_index(bucket_cols)["bucket_id"]
            .to_dict()
        )
        # ensure int ids
        id_map: dict = {}
        for k, v in base_map.items():
            # k is a tuple of bucket values
            id_map[tuple(k if isinstance(k, tuple) else (k,))] = int(v)
        next_id = (max(id_map.values()) + 1) if id_map else 0
    else:
        id_map = {}
        next_id = 0
        # assign ids to combos present in df to keep determinism
        for t in map(tuple, df_bucketed[bucket_cols].drop_duplicates().itertuples(index=False, name=None)):
            if t not in id_map:
                id_map[t] = next_id
                next_id += 1

    # 2) extend mapping to all keys in `counts.index` (including zero-available buckets)
    for t in map(tuple, counts.index):
        if t not in id_map:
            id_map[t] = next_id
            next_id += 1

    # helper: pretty desc from bucket tuple
    def _desc_from_tuple(tup: tuple) -> str:
        return " | ".join(f"{c}={v}" for c, v in zip(bucket_cols, tup))

    # ---------- targets per bucket ----------
    B = int(len(counts))
    base = total_size // B
    rem = total_size % B

    targets = pd.Series(base, index=counts.index, dtype=int)
    if rem > 0:
        add_idx = counts.sort_values(ascending=False).index[:rem]
        targets.loc[add_idx] += 1

    # ---------- draw samples & build gaps ----------
    sampled_frames = []
    gap_rows = []

    for bucket_key, target in targets.items():
        # mask for this bucket
        mask = (df_bucketed[bucket_cols] == pd.Series(bucket_key, index=bucket_cols)).all(axis=1)
        pool = df_bucketed.loc[mask]
        available = int(len(pool))
        k = int(min(target, available))

        # actual sampling (without replacement)
        if k > 0:
            sampled_frames.append(pool.sample(n=k, replace=False, random_state=random_state))

        # stable id/desc (even if available==0)
        key_tuple = tuple(bucket_key)
        bucket_id = id_map[key_tuple]
        bucket_desc = _desc_from_tuple(key_tuple)

        gap_rows.append({
            "bucket_id": bucket_id,
            "bucket_desc": bucket_desc,
            "target": int(target),
            "available": available,
            "sampled": k,
            "shortfall": max(int(target) - k, 0),
        })

    sampled = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else df_bucketed.iloc[0:0].copy()
    gaps = pd.DataFrame(gap_rows).sort_values(["shortfall", "target"], ascending=[False, False]).reset_index(drop=True)

    return sampled, gaps


def oversample_to_min_per_bucket(
    df: pd.DataFrame,
    bucket_cols: list,
    min_size: int = 30,
    repo_col: str = "repo",
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    For each bucket (defined by bucket_cols), if group size < min_size,
    randomly sample rows *with replacement* from that group to pad up to min_size.
    Duplicated rows keep all feature values but get a unique repo name.

    Returns a new DataFrame with an extra column `is_oversampled` (0/1).
    """
    if not bucket_cols:
        raise ValueError("bucket_cols must not be empty for oversampling.")

    rng = np.random.RandomState(random_state)
    parts = [df.copy()]
    parts[0]["is_oversampled"] = 0

    # Maintain global uniqueness for repo names
    existing = set(df[repo_col].astype(str)) if repo_col in df.columns else set()
    uid = count(1)

    def _unique_repo_name(base: str) -> str:
        # generate globally unique name
        candidate = f"{base}__dup{next(uid)}"
        while candidate in existing:
            candidate = f"{base}__dup{next(uid)}"
        existing.add(candidate)
        return candidate

    for _, g in df.groupby(bucket_cols, dropna=False, as_index=False):
        k = len(g)
        if k >= min_size:
            continue
        need = min_size - k

        # sample indices with replacement within this bucket
        sample_idx = rng.choice(g.index.values, size=need, replace=True)
        dup = df.loc[sample_idx].copy()

        # mark and rename repos to unique synthetic ids
        dup["is_oversampled"] = 1
        if repo_col in dup.columns:
            # prefer to keep a hint of origin; if repo缺失就用 bucket key
            base_names = dup[repo_col].astype(str).fillna("synthetic")
            dup[repo_col] = [ _unique_repo_name(b) for b in base_names ]
        parts.append(dup)

    out = pd.concat(parts, ignore_index=True)
    if "is_oversampled" not in out.columns:
        out["is_oversampled"] = 0
    return out