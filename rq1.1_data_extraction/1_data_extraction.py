# data_extraction.py
from __future__ import annotations

import logging
import sys

sys.stdout.reconfigure(line_buffering=True) 
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from services.extraction_helper import detect_ci_type  # CI auto-detection
from services.extraction_helper import (collect_commits_local, ensure_dir,
                                        normalize_repo_full_name,
                                        save_github_action_runs, save_issues,
                                        save_prs, save_releases,
                                        save_repo_source, save_travis_jobs)

from config.load_config import load_config

# ============================ simple config ============================
# Choose repo source:
#   "csv"     → read from base.csv & repo_quota_gaps.csv
#   "dataset" → scan dataset_dir/owner/repo
SOURCE_MODE = "csv"

# ---------------------------- configuration ----------------------------
cfg = load_config()

GITHUB_TOKENS: Optional[list] = cfg.github.tokens
TODAY: str = str(cfg.date.today)
TRAVIS_TOKENS: Optional[list] = cfg.travis.tokens

RQ1_DATA_PATH = Path(str(cfg.path.rq1_path))
dataset_dir = Path(str(cfg.path.dataset_path))
dataset_info_dir = Path(str(cfg.path.dataset_info_path))

# "base.csv" quota.csv""

all_metadata = RQ1_DATA_PATH / "quota.csv"
metadata_gap_path = RQ1_DATA_PATH / "quota.csv"

# ---------------------------- logging ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---------------------------- helpers ----------------------------
REQUIRED_COLUMNS = [
    "repo", "defaultBranch", "size", "releases", "commits", "watchers", "stargazers",
    "contributors", "createdAt", "totalIssues", "openIssues", "mainLanguage",
    "lastCommit", "totalPullRequests", "ci_type", "ci_adoption_time",
    "avg_contribution", "open_issue_ratio", "issues_per_day", "commits_per_day",
    "commits_per_day_bucket", "issues_per_day_bucket", "ci_type_bucket",
    "mainLanguage_bucket", "bucket_id", "bucket_desc",
]


def _read_repo_list(paths: List[Path]) -> pd.DataFrame:
    """Read repos from CSV, keep repo/defaultBranch/ci_type."""
    dfs = []
    for p in paths:
        if not p.exists():
            logging.warning(f"[warn] file not found: {p}")
            continue
        df = pd.read_csv(p, low_memory=False)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logging.warning(f"[warn] missing cols in {p.name}: {missing}")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["repo", "defaultBranch", "ci_type"])

    out = pd.concat(dfs, ignore_index=True)
    for col in ["repo", "defaultBranch", "ci_type"]:
        if col not in out.columns:
            out[col] = ""
    return out[["repo", "defaultBranch", "ci_type"]].dropna(how="all")


def _unique_repos(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Unique triples (repo, branch, ci_type)."""
    df = df.copy()
    for col in ["repo", "defaultBranch", "ci_type"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()
    df = df[df["repo"] != ""]
    return sorted(set((r.repo, r.defaultBranch, r.ci_type) for r in df.itertuples(index=False)))


def _repos_from_dataset_dir(root: Path) -> List[Tuple[str, str, str]]:
    """Scan dataset_dir/owner/repo and return triples (repo, '', '')."""
    repos: List[Tuple[str, str, str]] = []
    if not root.exists():
        logging.warning(f"[warn] dataset_dir not found: {root}")
        return repos
    for owner_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for repo_dir in sorted([p for p in owner_dir.iterdir() if p.is_dir()]):
            repos.append((f"{owner_dir.name}/{repo_dir.name}", "", ""))
    return repos


# ---------------------------- main entry ----------------------------
def downloading_data_each_repo(overwrite: bool = False) -> None:
    """
    Download data for each repo:
    - Source repos from CSV or dataset_dir (SOURCE_MODE).
    - Clone repo source into dataset_dir.
    - Save PRs, issues, releases into dataset_info_dir.
    - Save CI runs (GitHub Actions/Travis).
    - Save local commits.
    """
    ensure_dir(dataset_dir)
    ensure_dir(dataset_info_dir)

    mode = SOURCE_MODE if SOURCE_MODE in {"csv", "dataset"} else "csv"
    if mode != SOURCE_MODE:
        logging.warning(f"[warn] invalid SOURCE_MODE={SOURCE_MODE}, fallback 'csv'")
    logging.info(f"[info] SOURCE_MODE={mode}")

    repos = _repos_from_dataset_dir(dataset_dir) if mode == "dataset" else _unique_repos(
        _read_repo_list([all_metadata, metadata_gap_path])
    )

    if not repos:
        logging.info("[info] no repos found")
        return
    logging.info(f"[info] total repos: {len(repos)}")

    for repo_full_name, default_branch, ci_type in repos:
        print(f"\n---> {repo_full_name} (start)", flush=True)
        repo_full_name = normalize_repo_full_name(repo_full_name)
        if not repo_full_name or "/" not in repo_full_name:
            logging.warning(f"[skip] invalid repo: {repo_full_name}")
            continue

        owner, name = repo_full_name.split("/", 1)
        repo_src_dir = dataset_dir / owner / name
        repo_info_dir = dataset_info_dir / owner / name
        ensure_dir(repo_src_dir)
        ensure_dir(repo_info_dir)

        # --- clone / update source (note if using existing local copy)
        existed_before = repo_src_dir.exists() and any(repo_src_dir.iterdir())
        print(f"[source] cloning/updating into {repo_src_dir} (overwrite={overwrite})", flush=True)
        try:
            ok = save_repo_source(repo_full_name, repo_src_dir, overwrite=overwrite)
            if ok:
                suffix = "  # existing local copy" if existed_before else ""
                print(f"[source] ok: {repo_src_dir}{suffix} (repo={repo_full_name})")
            else:
                logging.warning(f"[source] skipped/failed: {repo_src_dir} (repo={repo_full_name})")
        except Exception as e:
            logging.exception(f"[source] failed: {repo_full_name} - {e}")

        # --- decide CI type (prefer provided, else detect from repo contents)
        ci = (ci_type or "").strip()
        if ci not in {"GitHubActions", "TravisCI"}:
            try:
                detected = detect_ci_type(repo_src_dir)
                ci = detected if detected in {"GitHubActions", "TravisCI"} else ""
            except Exception as e:
                logging.exception(f"[ci-detect] failed: {repo_full_name} - {e}")
                ci = ci  # keep whatever we had

        # --- banner (single CI status line)
        ci_label = ci if ci in {"GitHubActions", "TravisCI"} else "unknown"

        # --- metadata
        try:
            print(f"[prs.csv] count={save_prs(GITHUB_TOKENS, repo_full_name, repo_info_dir, overwrite)}", flush=True)
        except Exception as e:
            logging.exception(f"[prs] failed: {repo_full_name} - {e}")

        try:
            print(f"[issues.csv] count={save_issues(GITHUB_TOKENS, repo_full_name, repo_info_dir, overwrite)}", flush=True)
        except Exception as e:
            logging.exception(f"[issues] failed: {repo_full_name} - {e}")

        try:
            print(f"[releases.csv] count={save_releases(GITHUB_TOKENS, repo_full_name, repo_info_dir, overwrite)}", flush=True)
        except Exception as e:
            logging.exception(f"[releases] failed: {repo_full_name} - {e}")

        # --- CI runs (execution lines are tagged as ci-run)
        # if ci == "GitHubActions":
        #     try:
        #         print("[ci-run] GitHub Actions")
        #         print(f"[gactions.csv] count={save_github_action_runs(GITHUB_TOKENS, repo_full_name, repo_info_dir, overwrite)}")
        #     except Exception as e:
        #         logging.exception(f"[actions] failed: {repo_full_name} - {e}")

        # elif ci == "TravisCI":
        #     try:
        #         print("[ci-run] Travis")
        #         print(f"[jobs.csv] count={save_travis_jobs(TRAVIS_TOKENS, repo_full_name, repo_info_dir, overwrite)}")
        #     except Exception as e:
        #         logging.exception(f"[travis] failed: {repo_full_name} - {e}")

        # else:
        print("[ci-run] → try both")
        try:
            print(f"[gactions.csv] count={save_github_action_runs(GITHUB_TOKENS, repo_full_name, repo_info_dir, overwrite)}", flush=True)
        except Exception as e:
            logging.exception(f"[actions] failed: {repo_full_name} - {e}")
        try:
            print(f"[jobs.csv] count={save_travis_jobs(TRAVIS_TOKENS, repo_full_name, repo_info_dir, overwrite)}", flush=True)
        except Exception as e:
            logging.exception(f"[travis] failed: {repo_full_name} - {e}")

        # --- local commits
        try:
            n_commits = collect_commits_local(
                repo_src_dir=repo_src_dir,
                output_csv=repo_info_dir / "commits_local.csv",
                default_branch=(default_branch or "HEAD"),
                overwrite=overwrite,
            )
            print(f"[commits_local.csv] count={n_commits}")
        except Exception as e:
            logging.exception(f"[commits_local] failed: {repo_full_name} - {e}")


if __name__ == "__main__":
    downloading_data_each_repo(overwrite=False)