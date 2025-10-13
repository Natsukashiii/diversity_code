from __future__ import annotations

from pathlib import Path

import pandas as pd
from services.filter_rules import (apply_filters, enrich_metrics,
                                   merge_old_data, tidy_output,
                                   validate_columns)

from config.helper import summarize_csv
from config.load_config import load_config


def create_metadata_file(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Read raw dataset, apply filter rules, and output metadata.

    Filter rules applied:
    1. Project must not be archived, disabled, or locked.
    2. Must have at avg 1 release/issue/commits per year.
    5. Main language must be Java or Python.
    6. Last commit must be in 2024.
    7. CI adoption: at least one of TravisCI or GitHubActions must exist.
       - ci_type = "TravisCI" or "GitHubActions"
       - ci_adoption_time = corresponding adoption date

    Output dataset keeps only required metadata fields.
    """
    df = pd.read_csv(input_path)
    validate_columns(df)

    # Step 1: filter rows + resolve ci_type / ci_adoption_time
    filtered = apply_filters(df)

    # Step 2: add derived fields:
    #    - commits_per_year
    #    - open_issue_ratio
    enriched = enrich_metrics(filtered)

    # Step2.1: add old data
    enriched = merge_old_data(
    enriched,
    old_data_path= Path(cfg.path.rq1_path) / "extracted_data.csv",
    on="repo",
    keep_cols=["avg_issues_close_time", "size_per_dev"],  # None means all columns
    allow_overwrite=True
)


    # Step 3: keep only OUTPUT_COLUMNS (which already include the new fields)
    result = tidy_output(enriched)

    # Step 4: summarize and save
    summarize_csv(result)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"[OK] input={len(df)} kept={len(result)} → {out}")
    return result


if __name__ == "__main__":
    cfg = load_config()
    input_path = str(cfg.path.original_data)
    output_path = str(cfg.path.metadata)
    create_metadata_file(input_path, output_path)