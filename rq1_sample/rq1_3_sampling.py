# rq1_sampling.py
from __future__ import annotations

from pathlib import Path
from pprint import pprint

import pandas as pd
from services.sample_helper import (apply_buckets,
                                    attach_ranges_to_bucket_info,
                                    build_sampling_config,
                                    build_sampling_config_value,
                                    print_and_save_simple_counts,
                                    print_bucket_overview, print_quota_gaps,
                                    translate_desc_to_name, visualize)
from services.sample_method import (quota_sample, random_sample,
                                    stratified_sample)

from config.helper import load_csv, safe_to_csv
from config.load_config import load_config

# ---------------------------- adjust configuration ----------------------------

# Fields used to build buckets (numeric → quantile bins; categorical → unique) issues_per_day  commits_per_day  avg_contribution  open_issue_ratio project_age_days  avg_issues_close_time  test_percentage
# repo,defaultBranch,size,releases,commits,watchers,stargazers,contributors,createdAt,totalIssues,openIssues,mainLanguage,lastCommit,totalPullRequests,ci_type,ci_adoption_time,avg_contribution,open_issue_ratio,contributors,commits_per_day,project_age_days,size_per_dev
SAMPLE_FIELDS = ["open_issue_ratio", "project_age_days", "ci_type", "mainLanguage"]
#!!! currently sample SAMPLE_FIELDS = ["avg_issues_close_time", "project_age_days", "ci_type", "mainLanguage"]
# Fields used for visualization/diagnostics (fallback to SAMPLE_FIELDS if empty)
SHOW_FIELDS = ["commits", "totalIssues", "contributors", "stargazers"] or SAMPLE_FIELDS

# ---------------------------- configuration in config.yaml ----------------------------
cfg = load_config()
metadata_clean_path: str = str(cfg.path.metadata_cleaned)
metadata_raw_path: str = str(cfg.path.metadata)
output_quota_path: str = str(cfg.path.quota_path)
output_stratified_path: str = str(cfg.path.stratified_path)
output_random_path: str = str(cfg.path.random_path)
SAMPLE_SIZE = cfg.rq1.sample_size
RANDOM_STATE = cfg.rq1.random_state
NUMERIC_BIN_COUNT = cfg.rq1.numeric_bins
RQ1_DATA_PATH = Path(str(cfg.path.rq1_path))    
# ----------------------------------------------------------------------


def main() -> None:
    # Step 1. Load metadata
    df = load_csv(metadata_clean_path) 
    print(f"[INFO] Loaded metadata_clean_path: {len(df)} rows ← {metadata_clean_path}")

    # Step 2. Build sampling config and inspect (percentage mode: 30% - 70% , numeric mode: 2 bins)
    cfg = build_sampling_config_value (df, sample_fields=SAMPLE_FIELDS,mode="both")
    cfg_percentage = cfg["percentage"]
    cfg_numeric = cfg["numeric"]
    print("\n[Sampling Config]")
    pprint(cfg, sort_dicts=False)

 
    # Step 3 Apply buckets → add *_bucket columns + bucket_id; collect bucket catalog
    bucketed_map, bucket_info = apply_buckets(df, sampling_configs=[("numeric", cfg_numeric), ("percentage", cfg_percentage)])
    df_bucketed_numeric = bucketed_map["numeric"]
    df_bucketed_percentage = bucketed_map["percentage"]
    
    # Step 3.0 Generate the df_base
    safe_to_csv(df_bucketed_numeric, RQ1_DATA_PATH / "base.csv")    


    # Step 3.1 Save bucket catalog with human-friendly labels (range + low/high)
    bucket_cfg_path = Path(RQ1_DATA_PATH) / "bucket_config.csv"
    bucket_info["bucket_name"] = bucket_info.apply(lambda r: translate_desc_to_name(r), axis=1)
    bucket_info = attach_ranges_to_bucket_info(
        bucket_info=bucket_info,
        cfg_numeric=cfg_numeric,
        cfg_percentage=cfg_percentage,
        fields=["open_issue_ratio", "project_age_days"],
    )
    safe_to_csv(bucket_info, bucket_cfg_path)
    print(f"[INFO] Bucket config saved → {bucket_cfg_path}")

    # Step 3.2 Quick overview
    print_bucket_overview(df_bucketed_numeric, sample_fields=SAMPLE_FIELDS)

    # Step 4 Run sampling methods
    # Step 4.1 Random baseline (not persisted)
    df_random = random_sample(df_bucketed_numeric, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"\n[Random] sampled = {len(df_random)}")
    safe_to_csv(df_random, output_random_path)
    print_bucket_overview(df_random, sample_fields=SAMPLE_FIELDS, title="[Random] bucket counts")

    
    # Step 4.2 Stratified sampling (proportional to bucket sizes)
    df_stratified = stratified_sample(
        df_bucketed=df_bucketed_numeric,
        sample_size=SAMPLE_SIZE,
        bucket_cols=[f"{f}_bucket" for f in cfg_numeric.keys()],
        random_state=RANDOM_STATE,
    )
    safe_to_csv(df_stratified, output_stratified_path)
    print_bucket_overview(df_stratified, sample_fields=SAMPLE_FIELDS, title="[Stratified] bucket counts")

    # Step 4.3 Quota sampling (equal target per bucket; short buckets filled as much as possible)
    df_quota, gaps = quota_sample(
        df_bucketed=df_bucketed_numeric,
        bucket_cols=[f"{f}_bucket" for f in cfg_numeric.keys()],
        total_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )
    safe_to_csv(df_quota, output_quota_path)
    print_bucket_overview(df_quota, sample_fields=SAMPLE_FIELDS, title="[Quota] bucket counts")

    print_bucket_overview(df_bucketed_numeric, sample_fields=SAMPLE_FIELDS, title="[Base] bucket counts")


    # Step 6 Report summaries
    print_quota_gaps(gaps)
    safe_to_csv(gaps, RQ1_DATA_PATH / "quota_gaps.csv")

        # ---------- Dimension-level summaries (e.g., total Java, total TravisCI) ----------
    datasets = {
        "Base": df_bucketed_numeric,
        "Stratified": df_stratified,
        "Quota": df_quota,
        "Random": df_random,
    }
    for name, dfx in datasets.items():
        print_and_save_simple_counts(name=name, df=dfx, out_dir=RQ1_DATA_PATH)


    # Step 7 Visual diagnostics (ECDF + violin for SHOW_FIELDS)
    visualize(
        dfs=[df_bucketed_numeric, df_quota],
        labels=["Stratified", "Quota"],
        columns=SAMPLE_FIELDS,
        outpath=RQ1_DATA_PATH / "rq1_dataset_compare.pdf",
    )


if __name__ == "__main__":
    main()