# Outline
The following are the descriptions for the modules.
- **RQ0 – Data Preparation:**  
  Build initial metadata from raw dataset (CSV or directory) and apply filtering rules (enough analysis data).
- **RQ0.1 – Metadata Cleaning:**  
  Validate and clean metadata (remove invalid repos, inconsistent timestamps, missing files).
- **RQ1 – Sampling:**  
  Generate representative subsets (random, stratified, quota) from cleaned metadata for analysis.
- **RQ2 – Distribution Analysis:**  
  Compare activity metrics (commits, issues, PRs, etc.) between stratified and quota datasets.
- **RQ3 – RDD Analysis & Visualization:**  
  Data analysis on monthly metrics and visualization.


# RQ0 DATA PREPARE

## Overview
This module is for **creating the metadata file** used in the following sampling and analysis.  

The pipeline:
1. Read raw dataset (CSV or dataset directory, configured in `config.yaml`)
2. Apply filtering rules
3. Keep only required metadata fields
4. Output `metadata.csv`
5. Print

---

## Run

### Option 0: Generate metadata from raw dataset (recommended)
1. **Prepare input source** in `config.yaml`:
   - `csv_path`: Path to a CSV file containing repo metadata  
     - Required field: `"repo"` (format: `owner/rpo`)  
   - `dataset_dir`: If using a dataset directory, repos will be under `dataset/owner/repo`  
   - If both CSV and dataset_dir are provided, **CSV will be used by default**

2. **Run the script**
3. 
4. **Check the results**
    - Output file: {cfg.path.metadata} (default: data/rq1/metadata.csv) 
    - Summary statistics are printed in the console



### Option 1: Run with CSV (extracted data)
Replace data/dataset_50k.csv (or set cfg.path.original_data in config.yaml) with your own dataset (original), then rerun the script.
Make sure your CSV contains the required fields (or replace with your own field, but you need to correct the corresponding field name in services.filter_rules).

1. **input dataset** (CSV or directory) must contain the following fields:

```text
| Field           | Type      | Description                                                                 |
|-----------------|-----------|-----------------------------------------------------------------------------|
| repo            | string    | Repository name in `owner/repo` format                                      |
| isFork          | bool      | Whether the repo is a fork                                                  |
| isArchived      | bool      | Whether the repo is archived                                                |
| isDisabled      | bool      | Whether the repo is disabled                                                |
| isLocked        | bool      | Whether the repo is locked                                                  |
| defaultBranch   | string    | Default branch name                                                         |
| releases        | int       | Number of releases                                                          |
| commits         | int       | Total number of commits                                                     |
| watchers        | int       | Watcher count                                                               |
| stargazers      | int       | Stargazer count                                                             |
| contributors    | int       | Contributor count                                                           |
| createdAt       | datetime  | Project creation date                                                       |
| totalIssues     | int       | Total number of issues                                                      |
| openIssues      | int       | Number of open issues                                                       |
| mainLanguage    | string    | Main programming language                                                   |
| lastCommit      | datetime  | Timestamp of the last commit                                                |
| totalPullRequests | int     | Total number of pull requests                                               |
| ci_adoption_times | JSON-like | CI adoption info, e.g. `{'TravisCI': None, 'GitHubActions': '2019-09-12T17:33:21+02:00', ...}` |
```


2. **Filter Rules**
A project is kept only if all of the following hold:

```text
| Rule              | Condition |
|-------------------|-----------|
| **Repository state** | `isArchived = False` <br> `isDisabled = False` <br> `isLocked = False` |
| **Activity** | `releases > 12` <br> `commits / age_in_years > 12` <br> (`age_in_years` is computed from `createdAt` to now) <br> `totalIssues > 12` |
| **Main language**     | Must be `"Java"` or `"Python"` |
| **Recent activity**   | `lastCommit` is in **2024** |
| **CI adoption**       | At least one of **TravisCI** or **GitHubActions** in `ci_adoption_times` is not None <br> `ci_type = "TravisCI"` or `"GitHubActions"` <br> `ci_adoption_time = corresponding date` |
```
3. **Output Fields**
The final metadata file (`data/rq1/metadata.csv`) contains:
```text
| Field             | Description |
|-------------------|-------------|
| repo              | Repository name (`owner/repo`) |
| defaultBranch     | Default branch |
| releases          | Number of releases |
| commits           | Total commits |
| watchers          | Watchers count |
| stargazers        | Stargazers count |
| contributors      | Contributors count |
| createdAt         | Creation time |
| totalIssues       | Total issues |
| openIssues        | Open issues |
| openIssuesRatio   | Open issues ratios |
| mainLanguage      | Main language |
| lastCommit        | Last commit time |
| totalPullRequests | Total pull requests |
| ci_type           | `"TravisCI"` or `"GitHubActions"` |
| ci_adoption_time  | CI adoption timestamp |
```
















# RQ0.1  METADATA CREATION

## 3_data_cleaning.py

### Overview
This script cleans and validates repository metadata to ensure data quality for CI/CD analysis.  
It filters invalid repositories, inconsistent timestamps, and incomplete data files.

---

### Main Function
**`clean_metadata()`**  
- Normalizes repository names (`owner/repo`)  
- Parses and validates `createdAt` and `ci_adoption_time`  
- Applies filtering rules:
  1. Invalid or missing repo name  
  2. CI adoption time earlier than creation (because the adoption time is from commit history, it may come from a different branch )
  3. Missing CI adoption timestamp  
  4. Less than 1 year between creation → adoption or adoption → now  
  5. Missing required CSV files (`commits_local`, `prs`, `releases`, `issues`) or both CI logs are empty (`gactions`, `jobs`)

---

### Input
- **`metadata.csv`** (in `RQ1_DATA_PATH`)  
---
### Output
- **`metadata_cleaned.csv`**  (in `RQ1_DATA_PATH`)  
---



# RQ1: SAMPLING

## Overview
This script performs dataset sampling for RQ1 using multiple strategies to create representative subsets of repositories.  
It generates stratified, quota-based, and random samples from the cleaned metadata.

---

### Main Function
**`main()`**  
- Loads the cleaned metadata file (`metadata_cleaned.csv`)  
- Builds sampling configurations based on key project attributes  
- Applies bucketization (numeric & percentage bins)  
- Generates and saves different sampling datasets:
  1. **Base dataset** – bucketed full data  
  2. **Random sample** – simple random selection  
  3. **Stratified sample** – proportional sampling by buckets  
  4. **Quota sample** – balanced sampling with equal bucket targets  
- Saves sampling summaries, bucket info, and visualization for comparison

---

### Input
- **`metadata_cleaned.csv`** (from `RQ1_DATA_PATH`)  
  Includes fields such as:
  - `repo`, `createdAt`, `ci_type`, `mainLanguage`, `open_issue_ratio`, `project_age_days`, etc.

---

### Output
```text
| File | Description |
|------|--------------|
| **`base.csv`** | Full dataset with added bucket columns |
| **`random.csv`** | Randomly sampled subset |
| **`stratified.csv`** | Stratified subset proportional to bucket sizes |
| **`quota.csv`** | Quota-based balanced subset |
| **`bucket_config.csv`** | Bucket configuration and ranges |
| **`quota_gaps.csv`** | Buckets that could not reach quota targets |
| **`rq1_dataset_compare.pdf`** | Visual diagnostics (ECDF + violin plots) |
```
---


# RQ2&3  DATA ANALYSIS & PLOT

## data_extraction.py

### Overview
This script automates data collection for each GitHub repository in the study dataset.  
It clones repositories, extracts project metadata (PRs, issues, releases, commits),  
and saves CI/CD run data (GitHub Actions and Travis CI) into structured folders.

---

### Main Function
**`downloading_data_each_repo()`**   
- Downloads and saves:
  - **Pull requests →** `prs.csv`
  - **Issues →** `issues.csv`
  - **Releases →** `releases.csv`
  - **CI runs →** `gactions.csv`, `jobs.csv`
  - **Local commits →** `commits_local.csv`



## rq2_distribution.py

### Overview
Computes and compares monthly metrics between two datasets (“Stratified” from `base.csv` and “Quota” from `quota.csv`).  

---

### Inputs
- **`RQ1_DATA_PATH/base.csv`** — list of repositories for the Stratified dataset (must include a `repo` column).
- **`RQ1_DATA_PATH/quota.csv`** — list of repositories for the Quota dataset (must include a `repo` column).
- **Per-repo data directory** `DATASET_INFO_DIR/owner/repo/` containing extracted files used by `build_monthly_metrics_for_repo` (e.g., `commits_local.csv`, `issues.csv`, `prs.csv`, etc.).
- (Optional, if already computed)  
  - **`RQ2_DATA_PATH/rq2_metric_values.csv`** — long-form metric values to plot directly.  
  - **`RQ2_DATA_PATH/rq2_distribution.csv`** — existing stats summary 

---

### Outputs
- **`RQ2_DATA_PATH/rq2_distribution.csv`** — per-metric comparison table 
- **`RQ2_DATA_PATH/rq2_metric_values.csv`** — long-form values (`dataset, metric, value, repo, month`) for plotting.
- **`RQ2_DATA_PATH/rq2_distributions.pdf`** — histograms (overlap) of Stratified vs. Quota for each metric.

---


## rdd_rq3.py

### Overview
Runs RDD analyses on monthly repo metrics

### Inputs
- **`RQ1_DATA_PATH/base.csv`** — index for Stratified dataset (default when `input_source="stra"`).
- **`RQ1_DATA_PATH/quota.csv`** — index for Quota dataset (when `input_source="quota"`).
  - Expected columns (best effort): `repo`, `mainLanguage`, `ci_adoption_time`, `bucket_id`, `bucket_name`, `createdAt`, `commits`, `contributors`.
- **Per-repo data** under **`DATASET_INFO_DIR/owner/repo/`**:
  - `commits_local.csv`, `issues.csv`, `prs.csv`, `releases.csv` (used to build monthly metrics).

### Outputs
- RDD result tables (CSV):
  - **`RQ3_DATA_PATH/rdd_overall_stra.csv`** — STRA (base) overall.
  - **`RQ3_DATA_PATH/rdd_overall_quota.csv`** — QUOTA overall.
  - **`RQ3_DATA_PATH/rdd_by_bucket_stra.csv`** — STRA per-bucket.
  - (If running via `run_rq3_pipeline`, a custom path can be provided via `out_path`.)
- Each CSV includes coefficients (`time`, `intervention` level γ, `time_after` slope Δ), SE/p-values, Wald χ², R² (marginal/conditional), VIF, sample sizes, and derived trend direction.



## rq3_plot.py

### Overview
Generates publication-ready figures from RDD result tables:
---

### Inputs
- **`RQ3_DATA_PATH/rdd_by_bucket_stra.csv`**
- **`RQ3_DATA_PATH/rdd_overall_stra.csv`**
- **`RQ3_DATA_PATH/rdd_overall_quota.csv`**

> Paths are read from `config.load_config()`.

---

### Outputs
- Directory: **`RQ3_PLOT_PATH/plots/`**
  - **`<metric>.pdf`** — forest-style plot (effect size & 95% CI + repo count bar).
  - **`legend.pdf`** — standalone horizontal legend.
  - **`opposite_by_dimension.pdf`** — 100% stacked bars for contexts opposite to STRA.
  - **`by_sign_dimension.pdf`** — 2×4 grid of 100% stacked bars split by sign (Δ>0 / Δ<0).
  - *(Optional CSV exports for counts/percents are present in code but commented out.)*

---
