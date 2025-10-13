# extraction_helper.py
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import pandas as pd
import requests

from config.github_helper import request_github_tokens_pool

GITHUB_API_URL = "https://api.github.com"
TRAVIS_API_URL = "https://api.travis-ci.com"
_TRAVIS_UA = "rq-suite-travis/1.0"
# ---------------------------- base utils ----------------------------
def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def detect_ci_type(repo_src_dir: Path) -> str:
    """
    Detect CI type by repo contents.
    Returns: "GitHubActions" | "TravisCI" | ""
    Heuristics:
      - .github/workflows/*.yml|*.yaml → GitHubActions
      - .travis.yml → TravisCI
    If both exist, prefer GitHubActions.
    """
    try:
        if not repo_src_dir or not Path(repo_src_dir).exists():
            return ""

        workflows = Path(repo_src_dir) / ".github" / "workflows"
        has_actions = workflows.exists() and any(
            p.suffix.lower() in {".yml", ".yaml"} for p in workflows.glob("*")
        )

        has_travis = (Path(repo_src_dir) / ".travis.yml").exists()

        if has_actions:
            return "GitHubActions"
        if has_travis:
            return "TravisCI"
        return ""
    except Exception:
        return ""


def normalize_repo_full_name(repo: str) -> str:
    """Normalize 'owner/repo' string."""
    repo = (repo or "").strip().strip("/")
    # Keep original case for API correctness; filesystem dirs will use same.
    return repo

def _resp_to_json(resp) -> Optional[dict]:
    """Safe JSON parse."""
    if not resp:
        return None
    try:
        return resp.json()
    except Exception:
        return None

def _pool_get(
    tokens,
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    *,
    json: Optional[dict] = None,
    data: Optional[dict] = None,
    method: str = "GET",
):
    """Token-pooled request wrapper (GET by default)."""
    return request_github_tokens_pool(
        tokens,
        url=url,
        method=method,
        params=params or {},
        headers=headers or {},
        json=json,       # <-- correct kw
        data=data,       # <-- supported
        timeout=30,
    )

def _pool_paginated(tokens, url: str, per_page: int = 100, extra_params: Optional[dict] = None):
    """Yield paginated JSON arrays from GitHub REST v3."""
    page = 1
    while True:
        params = {"per_page": per_page, "page": page}
        if extra_params:
            params.update(extra_params)
        resp = _pool_get(tokens, url, params=params)
        if not resp or resp.status_code != 200:
            logging.warning(f"[api] request failed: {url} - status={getattr(resp, 'status_code', None)}")
            return
        data = _resp_to_json(resp)
        if not data:
            return
        # endpoints may return dict or list
        if isinstance(data, dict) and "workflow_runs" in data:
            items = data["workflow_runs"]
        elif isinstance(data, list):
            items = data
        else:
            # unknown shape
            items = []
        if not items:
            return
        yield items
        if len(items) < per_page:
            return
        page += 1
        time.sleep(0.3)

# ---------------------------- repo source ----------------------------
def save_repo_source(repo_full_name: str, repo_src_dir: Path, overwrite: bool = False) -> bool:
    """
    Clone or reuse repository source into repo_src_dir.

    Behavior:
    - If dir exists and NOT overwrite:
        * If it's a git repo and its 'origin' points to the same repo → reuse.
        * If origin mismatch / not a git repo / broken → delete and reclone.
    - If dir exists and overwrite=True → delete and reclone.
    - Otherwise → fresh clone.

    Returns True on success, False on failure.
    """
    def _read_origin(p: Path) -> str:
        """Try to read 'origin' remote URL; return '' on failure."""
        git_dir = p / ".git"
        if not git_dir.exists():
            return ""
        # 1) try `git remote get-url origin`
        try:
            out = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(p),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
        # 2) fallback to parse .git/config
        try:
            cfg_txt = (git_dir / "config").read_text(errors="ignore")
            return cfg_txt
        except Exception:
            return ""

    def _matches(repo_full: str, origin_text: str) -> bool:
        """
        Check whether origin_text matches target repo.
        Accept https and ssh forms; case-insensitive contains check.
        """
        if not origin_text:
            return False
        needle = f"github.com/{repo_full}".lower()
        text = origin_text.lower()
        # common forms:
        #   https://github.com/owner/name.git
        #   git@github.com:owner/name.git
        return (needle in text)

    # --- existing directory handling ---
    if repo_src_dir.exists() and any(repo_src_dir.iterdir()):
        if overwrite:
            shutil.rmtree(repo_src_dir, ignore_errors=True)
        else:
            origin = _read_origin(repo_src_dir)
            if _matches(repo_full_name, origin):
                # looks like the correct repo → reuse
                return True
            else:
                # mismatch / not a git repo / broken → reclone
                logging.warning(
                    f"[git] local dir '{repo_src_dir}' does not match target '{repo_full_name}', recloning..."
                )
                shutil.rmtree(repo_src_dir, ignore_errors=True)

    # --- fresh clone ---
    ensure_dir(repo_src_dir.parent)
    try:
        subprocess.run(
            ["git", "clone", "--no-tags", "--progress",
             f"https://github.com/{repo_full_name}.git", str(repo_src_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.warning(f"[git] clone failed: {repo_full_name} - {e}")
        return False
# ---------------------------- PRs ----------------------------
def save_prs(tokens, repo_full_name: str, repo_info_dir: Path, overwrite: bool = False) -> int:
    """
    Save all PRs into prs.csv. Keep all fields returned by API.
    """
    ensure_dir(repo_info_dir)
    out = repo_info_dir / "prs.csv"
    if out.exists() and not overwrite:
        try:
            return len(pd.read_csv(out, low_memory=False))
        except Exception:
            pass

    all_rows: List[dict] = []
    for items in _pool_paginated(tokens, f"{GITHUB_API_URL}/repos/{repo_full_name}/pulls", extra_params={"state": "all"}):
        all_rows.extend(items)

    if not all_rows:
        # TODO: for very large repos, consider GraphQL search or since filters with backfill.
        out.write_text("")  # create empty file
        return 0

    df = pd.json_normalize(all_rows)
    df.to_csv(out, index=False)
    return len(df)

# ---------------------------- Issues ----------------------------
def save_issues(tokens, repo_full_name: str, repo_info_dir: Path, overwrite: bool = False) -> int:
    """
    Save all issues (not PRs) into issues.csv. Keep all fields returned by API.
    """
    ensure_dir(repo_info_dir)
    out = repo_info_dir / "issues.csv"
    if out.exists() and not overwrite:
        try:
            return len(pd.read_csv(out, low_memory=False))
        except Exception:
            pass

    all_rows: List[dict] = []
    for items in _pool_paginated(tokens, f"{GITHUB_API_URL}/repos/{repo_full_name}/issues", extra_params={"state": "all"}):
        # GitHub /issues includes PRs; filter them out
        real_issues = [x for x in items if "pull_request" not in x]
        all_rows.extend(real_issues)

    if not all_rows:
        # TODO: consider GraphQL timeline for events if needed.
        out.write_text("")
        return 0

    df = pd.json_normalize(all_rows)
    df.to_csv(out, index=False)
    return len(df)

# ---------------------------- Releases ----------------------------
def save_releases(tokens, repo_full_name: str, repo_info_dir: Path, overwrite: bool = False) -> int:
    """
    Save all releases into releases.csv. Keep all fields returned by API.
    """
    ensure_dir(repo_info_dir)
    out = repo_info_dir / "releases.csv"
    if out.exists() and not overwrite:
        try:
            return len(pd.read_csv(out, low_memory=False))
        except Exception:
            pass

    all_rows: List[dict] = []
    for items in _pool_paginated(tokens,f"{GITHUB_API_URL}/repos/{repo_full_name}/releases", extra_params=None):
        all_rows.extend(items)

    if not all_rows:
        # TODO: consider fetching git tags for more complete "release frequency".
        out.write_text("")
        return 0

    df = pd.json_normalize(all_rows)
    df.to_csv(out, index=False)
    return len(df)

# ---------------------------- GitHub Actions ----------------------------
def save_github_action_runs(tokens, repo_full_name: str, repo_info_dir: Path, overwrite: bool = False) -> Optional[int]:
    """
    Save all workflow runs into gactions.csv. Keep all fields returned by API.
    Return None if endpoint not available (e.g., repo disabled Actions).
    """
    ensure_dir(repo_info_dir)
    out = repo_info_dir / "gactions.csv"
    if out.exists() and not overwrite:
        try:
            return len(pd.read_csv(out, low_memory=False))
        except Exception:
            pass

    # Check if actions is enabled; if 404, return None
    probe = _pool_get(
    tokens,
    f"{GITHUB_API_URL}/repos/{repo_full_name}/actions/runs",
    params={"per_page": 1, "page": 1},
)
    if not probe or probe.status_code == 404:
        # Actions disabled or no access
        return None
    if probe.status_code != 200:
        logging.warning(f"[actions] probe failed: {repo_full_name} - status={probe.status_code}")
        return None

    all_rows: List[dict] = []
    for items in _pool_paginated(tokens, f"{GITHUB_API_URL}/repos/{repo_full_name}/actions/runs"):
        all_rows.extend(items)

    if not all_rows:
        out.write_text("")
        return 0

    df = pd.json_normalize(all_rows)
    df["repo"] = repo_full_name
    df.to_csv(out, index=False)
    return len(df)


# ---------------------------- Travis CI (placeholder) ----------------------------

TRAVIS_API_URL = "https://api.travis-ci.com"
TRAVIS_ORG_URL =  "https://api.travis-ci.org"
_TRAVIS_UA = "rq-suite-travis/1.0"

def _travis_request(tokens: List[str], url: str, params: Optional[dict] = None, timeout: int = 30, *, debug: bool = False) -> requests.Response:
    """Travis v3 request with token rotation. Raise for non-2xx."""
    if not tokens:
        raise RuntimeError("No Travis tokens provided.")
    last_err = None
    for token in tokens:
        headers = {
            "Travis-API-Version": "3",
            "Authorization": f"token {token}",
            "User-Agent": _TRAVIS_UA,
            "Accept": "application/json",
        }
        try:
            if debug:
                logging.debug(f"[travis] GET {url} params={params}")
            resp = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
            if debug:
                logging.debug(f"[travis] status={resp.status_code} url={resp.url}")
                if resp.text:
                    # Show only first ~500 chars to avoid noise
                    logging.debug(f"[travis] body[:500]={resp.text[:500]}")
            # 429/403 rate-limited → try next token
            if resp.status_code in (429, 403) and "rate" in (resp.text or "").lower():
                last_err = RuntimeError(f"rate limited: {resp.status_code}")
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_err = e
            if debug:
                logging.debug(f"[travis] token failed: {e}")
            continue
    if last_err:
        raise last_err
    raise RuntimeError("All Travis tokens failed.")


# ------- helpers for save_travis_jobs -------
def _extract_job_row(repo_full_name: str, job: dict) -> dict:
    repo = job.get("repository") or {}
    commit = job.get("commit") or {}
    owner = job.get("owner") or {}

    return {
        "repo": repo_full_name,
        "job_id": job.get("id"),
        "job_number": job.get("number"),
        "job_state": job.get("state"),
        "job_started_at": job.get("started_at"),
        "job_finished_at": job.get("finished_at"),
        "allow_failure": job.get("allow_failure"),
        "queue": job.get("queue"),
        "stage": (job.get("stage") or {}).get("name") if isinstance(job.get("stage"), dict) else job.get("stage"),
        "vm_size": job.get("vm_size"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),

        # repository info
        "repository_id": repo.get("id"),
        "repository_slug": repo.get("slug"),
        "repository_name": repo.get("name"),

        # commit info
        "commit_id": commit.get("id"),
        "commit_sha": commit.get("sha"),
        "commit_ref": commit.get("ref"),
        "commit_message": commit.get("message"),
        "commit_committed_at": commit.get("committed_at"),
        "commit_compare_url": commit.get("compare_url"),

        # owner info
        "owner_id": owner.get("id"),
        "owner_login": owner.get("login"),
        "owner_name": owner.get("name"),
        "owner_vcs_type": owner.get("vcs_type"),
    }

def save_travis_jobs(
    tokens: List[str],
    repo_full_name: str,
    repo_info_dir: Path,
    overwrite: bool = False,
    *,
    debug: bool = False
) -> Optional[int]:
    """
    Fetch Travis jobs and save to jobs.csv.
    - Try .com first, then .org on 404.
    - Never raise to caller: print a short reason and return 0 (write empty file).
    """
    def _write_empty(out_path: Path) -> int:
        out_path.write_text("")
        return 0

    def _collect_with_base(base: str) -> List[dict]:
        """Collect jobs from one base (.com or .org). Raises on HTTP errors."""
        slug = quote(repo_full_name, safe="")
        next_url = f"{base}/repo/{slug}/builds"
        next_params = {"limit": 100, "include": "build.jobs", "sort_by": "started_at:desc"}

        rows: List[dict] = []
        while next_url:
            resp = _travis_request(tokens, next_url, params=next_params, debug=debug)
            data = resp.json() if resp is not None else {}

            builds = data.get("builds", []) or []
            top_jobs = data.get("jobs", []) or []

            # Index top-level jobs by build id (rarely needed)
            jobs_by_build: Dict[int, List[dict]] = {}
            for j in top_jobs:
                b = (j.get("build") or {})
                bid = b.get("id")
                if bid is not None:
                    jobs_by_build.setdefault(bid, []).append(j)

            for b in builds:
                bid = b.get("id")
                jobs = None
                if isinstance(b.get("jobs"), list) and b["jobs"]:
                    jobs = b["jobs"]
                if jobs is None and bid in jobs_by_build:
                    jobs = jobs_by_build[bid]
                if jobs is None and bid is not None:
                    # Fallback per-build jobs; keep base consistent
                    jresp = _travis_request(tokens, f"{base}/build/{bid}/jobs", debug=debug)
                    jdata = jresp.json() if jresp is not None else {}
                    jobs = jdata.get("jobs", []) or []

                for j in jobs or []:
                    row = _extract_job_row(repo_full_name, j)
                    row["build_id"] = bid
                    rows.append(row)

            # Pagination
            pag = data.get("@pagination") or {}
            if pag.get("is_last") is True or not pag.get("next"):
                next_url = None
                next_params = None
            else:
                href = (pag.get("next") or {}).get("@href")
                next_url = f"{base}{href}" if href else None
                next_params = None

        return rows

    # Prepare output
    repo_info_dir.mkdir(parents=True, exist_ok=True)
    out = repo_info_dir / "jobs.csv"
    if out.exists() and not overwrite:
        try:
            return len(pd.read_csv(out, low_memory=False))
        except Exception:
            pass

    # Try .com then .org, handling 404/other errors quietly.
    try:
        rows = _collect_with_base(TRAVIS_API_URL)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 404:
            # Try .org when .com says Not Found
            try:
                rows = _collect_with_base(TRAVIS_ORG_URL)
            except requests.HTTPError as e2:
                code2 = getattr(e2.response, "status_code", None)
                if code2 == 404:
                    # Neither .com nor .org has this repo
                    print(f"[travis] no builds for {repo_full_name} on .com/.org (404).")
                    return _write_empty(out)
                else:
                    print(f"[travis] error on .org for {repo_full_name}: HTTP {code2}.")
                    return _write_empty(out)
            except Exception as e2:
                print(f"[travis] unexpected error on .org for {repo_full_name}: {e2}.")
                return _write_empty(out)
        else:
            print(f"[travis] error on .com for {repo_full_name}: HTTP {code}.")
            return _write_empty(out)
    except Exception as e:
        print(f"[travis] unexpected error on .com for {repo_full_name}: {e}.")
        return _write_empty(out)

    # Save
    if not rows:
        print(f"[travis] empty result for {repo_full_name}.")
        return _write_empty(out)

    pd.DataFrame(rows).to_csv(out, index=False)
    return len(rows)
# ---------------------------- Local commits (default branch) ----------------------------
def _git(cmd: List[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Run git command (force UTF-8, tolerate bad bytes)."""
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",     
        errors="replace",     
    )

def _collect_git_log(repo_src_dir: Path) -> List[dict]:
    """
    Collect commit log with churn and file list.
    Output fields:
      commit_sha, author_name, author_email, author_date, parent_count, is_merge,
      insertions, deletions, files_changed, files_json
    """
    # Use --numstat to get additions/deletions and file names per commit
    pretty = "%H%x01%an%x01%ae%x01%ad%x01%P"
    cmd = [
    "git",
    "-c", "i18n.logOutputEncoding=UTF-8",  
    "log",
    "--encoding=UTF-8",                    
    "--date=iso-strict",
    f"--pretty=format:{pretty}",
    "--numstat",
]
    try:
        p = _git(cmd, cwd=repo_src_dir, check=True)
    except subprocess.CalledProcessError as e:
        logging.warning(f"[git] log failed: {repo_src_dir} - {e}")
        return []

    lines = p.stdout.splitlines()
    rows: List[dict] = []
    current: Optional[dict] = None

    def _flush_current():
        if current is None:
            return
        # Aggregate totals if missing
        current["insertions"] = current.get("insertions", 0)
        current["deletions"] = current.get("deletions", 0)
        current["files_changed"] = len(current.get("files", []))
        current["files_json"] = json.dumps(current.get("files", []), ensure_ascii=False)
        rows.append({
            "commit_sha": current["commit_sha"],
            "author_name": current["author_name"],
            "author_email": current["author_email"],
            "author_date": current["author_date"],
            "parent_count": current["parent_count"],
            "is_merge": 1 if current["parent_count"] > 1 else 0,
            "insertions": current["insertions"],
            "deletions": current["deletions"],
            "files_changed": current["files_changed"],
            "files_json": current["files_json"],
        })

    for ln in lines:
        if "\x01" in ln:
            # New commit header
            if current is not None:
                _flush_current()
            parts = ln.split("\x01")
            # parts: [sha, name, email, date, parents]
            parents = parts[4].strip().split() if len(parts) > 4 else []
            current = {
                "commit_sha": parts[0].strip(),
                "author_name": parts[1].strip() if len(parts) > 1 else "",
                "author_email": parts[2].strip() if len(parts) > 2 else "",
                "author_date": parts[3].strip() if len(parts) > 3 else "",
                "parent_count": len(parents),
                "files": [],
                "insertions": 0,
                "deletions": 0,
            }
        else:
            # numstat or blank
            if current is None:
                continue
            ln = ln.strip()
            if not ln:
                continue
            # numstat format: <additions>\t<deletions>\t<path>
            parts = ln.split("\t")
            if len(parts) >= 3:
                add, dele, path = parts[0], parts[1], "\t".join(parts[2:])
                try:
                    a = 0 if add == "-" else int(add)
                except Exception:
                    a = 0
                try:
                    d = 0 if dele == "-" else int(dele)
                except Exception:
                    d = 0
                current["insertions"] += a
                current["deletions"] += d
                current["files"].append(path)

    if current is not None:
        _flush_current()

    return rows


def _run_git(repo: Path, *args: str) -> str:
    out = subprocess.run(
    ["git", *args],
    cwd=str(repo),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",   
    errors="replace",   
    check=False,
)
    return out.stdout.strip()

def _ensure_full_history(repo: Path) -> None:
    # If shallow, unshallow
    try:
        is_shallow = _run_git(repo, "rev-parse", "--is-shallow-repository")
        if is_shallow.lower() == "true":
            _run_git(repo, "fetch", "--unshallow", "--tags")
    except Exception:
        # Fallback if --unshallow not supported
        _run_git(repo, "fetch", "--depth=2147483647", "--tags")
    # Make sure all remote branches are present
    _run_git(repo, "fetch", "origin", "+refs/heads/*:refs/remotes/origin/*", "--prune")

def _resolve_default_branch(repo: Path) -> Optional[str]:
    # Try origin/HEAD → refs/remotes/origin/<branch>
    ref = _run_git(repo, "symbolic-ref", "refs/remotes/origin/HEAD")
    if ref.startswith("refs/remotes/origin/"):
        return ref.split("/")[-1]
    # Fallback: parse `git remote show origin`
    remote_show = _run_git(repo, "remote", "show", "origin")
    for line in remote_show.splitlines():
        line = line.strip()
        if line.lower().startswith("head branch:"):
            return line.split(":", 1)[1].strip()
    # Last resort: current branch name (may be 'HEAD' if detached)
    curr = _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    return None if curr == "HEAD" else curr

def _checkout_local_tracking(repo: Path, branch: str) -> None:
    # Prefer local branch; if missing, create tracking branch from remote
    local_branches = set(_run_git(repo, "branch", "--list").replace("*", "").split())
    if branch in local_branches:
        _run_git(repo, "checkout", branch)
        return
    # If remote exists, create local tracking branch
    rem = f"origin/{branch}"
    rem_list = _run_git(repo, "branch", "-r", "--list", rem)
    if rem_list:
        _run_git(repo, "checkout", "-B", branch, rem)
        return
    # Fall back
    _run_git(repo, "checkout", branch)

def collect_commits_local(
    repo_src_dir: Path,
    output_csv: Path,
    default_branch: str = "HEAD",
    overwrite: bool = False,
) -> int:
    """
    Save default-branch commit history into commits_local.csv (full history).
    """
    if output_csv.exists() and not overwrite:
        try:
            return len(pd.read_csv(output_csv, low_memory=False))
        except Exception:
            pass

    if not repo_src_dir.exists():
        output_csv.write_text("")
        return 0

    # 1) ensure full history is available
    try:
        _ensure_full_history(repo_src_dir)
    except Exception:
        # non-fatal; continue with whatever history exists
        pass

    # 2) resolve real default branch if default is unknown/HEAD
    branch = (default_branch or "").strip()
    if branch.upper() == "HEAD" or branch == "":
        resolved = _resolve_default_branch(repo_src_dir)
        if resolved:
            branch = resolved

    # 3) checkout the target branch (create tracking branch if needed)
    try:
        target = branch if branch else "HEAD"
        _checkout_local_tracking(repo_src_dir, target)
    except Exception:
        # non-fatal; continue on current HEAD
        pass

    # 4) collect logs
    rows = _collect_git_log(repo_src_dir)  # your existing collector
    if not rows:
        output_csv.write_text("")
        return 0

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return len(df)