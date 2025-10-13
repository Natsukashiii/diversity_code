import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SourceCfg:
    original_data: Optional[Path] = None
    dataset_dir: Optional[Path] = None
    repo_field: str = "repo"

@dataclass
class Config:
    server: bool = False

@dataclass
class DateCfg:
    today: str = "2025-04-02"


@dataclass
class PathCfg:
    rq1_path: Optional[Path] = None
    rq2_path: Optional[Path] = None
    rq3_data_path: Optional[Path] = None
    rq3_plot_path: Optional[Path] = None
    original_data: Optional[Path] = None
    metadata: Optional[Path] = None
    metadata_cleaned: Optional[Path] = None
    quota_path: Optional[Path] = None
    stratified_path: Optional[Path] = None
    random_path: Optional[Path] = None
    dataset_path: Optional[Path] = None
    dataset_info_path: Optional[Path] = None


@dataclass
class Rq1Cfg:
    # sample_size: int = 480
    sample_size: int = 480
    random_state: int = 42
    numeric_bins: int = 2


@dataclass
class GithubCfg:
    tokens_str: Optional[str] = None
    tokens: List[str] = field(default_factory=list)


@dataclass
class TravisCfg:
    tokens_str: Optional[str] = None
    tokens: List[str] = field(default_factory=list)


@dataclass
class AppCfg:
    source: SourceCfg
    path: PathCfg
    github: GithubCfg
    rq1: Rq1Cfg
    date: DateCfg
    travis: TravisCfg
    config: Config
    raw: Dict[str, Any] = field(default_factory=dict)


# ----------------- helpers -----------------
def resolve_relative(base_dir: Path, maybe_path: Optional[str]) -> Optional[Path]:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _split_tokens(token_str: Optional[str]) -> List[str]:
    if not token_str:
        return []
    parts = [t.strip() for t in token_str.split(",")]
    return [t for t in parts if t]


# ----------------- main -----------------
def load_config(path: str = "config.yaml", use_server: bool = False) -> AppCfg:
    """Load config.yaml. If use_server=True, use server.path instead of path."""
    candidates = [
        path,
        os.getenv("CONFIG_PATH"),
        "/workspace/config.yaml",
        "config.yaml",
    ]

    chosen = None
    for p in candidates:
        if not p:
            continue
        rp = Path(p).resolve()
        if rp.exists() and rp.is_file():
            chosen = rp
            break

    if chosen is None:
        tried = [str(Path(p).resolve()) for p in candidates if p]
        raise FileNotFoundError(
            "[load_config] config.yaml not found.\nTried:\n" + "\n".join(f"  - {t}" for t in tried)
        )

    with open(chosen, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_dir = chosen.parent

     # Config
    config_raw = raw.get("config", {})
    config_cfg = Config(server=config_raw.get("server", False))

    # Source
    src_raw = raw.get("source", {})
    source = SourceCfg(
        original_data=resolve_relative(base_dir, src_raw.get("original_data")),
        dataset_dir=resolve_relative(base_dir, src_raw.get("dataset_dir")),
        repo_field=src_raw.get("repo_field", "repo"),
    )

    # Path (choose local path or server.path depending on flag)
    use_server = config_cfg.server
    key = "server.path" if use_server else "path"
    path_raw = raw.get(key, {})
    path_cfg = PathCfg(
        rq1_path=resolve_relative(base_dir, path_raw.get("rq1_path")),
        rq2_path=resolve_relative(base_dir, path_raw.get("rq2_path")),
        rq3_data_path=resolve_relative(base_dir, path_raw.get("rq3_data_path")),
        rq3_plot_path=resolve_relative(base_dir, path_raw.get("rq3_plot_path")),
        original_data=resolve_relative(base_dir, path_raw.get("original_data_filename")),
        metadata=resolve_relative(base_dir, path_raw.get("metadata_filename")),
        metadata_cleaned=resolve_relative(base_dir, path_raw.get("metadata_cleaned_filename")),
        quota_path=resolve_relative(base_dir, path_raw.get("quota_filename")),
        stratified_path=resolve_relative(base_dir, path_raw.get("stratified_filename")),
        random_path=resolve_relative(base_dir, path_raw.get("random_filename")),
        dataset_path=resolve_relative(base_dir, path_raw.get("dataset_path")),
        dataset_info_path=resolve_relative(base_dir, path_raw.get("dataset_info_path")),
    )

    # Rq1
    rq1_raw = raw.get("rq1", {})
    rq1_cfg = Rq1Cfg(
        sample_size=rq1_raw.get("sample_size", 480),
        random_state=rq1_raw.get("random_state", 42),
        numeric_bins=rq1_raw.get("numeric_bins", 2),
    )

    # Github
    github_raw = raw.get("github", {})
    tokens_str_raw = github_raw.get("tokens_str")
    github_cfg = GithubCfg(
        tokens_str=tokens_str_raw,
        tokens=_split_tokens(tokens_str_raw),
    )

    # Travis
    travis_raw = raw.get("travis", {})
    travis_cfg = TravisCfg(
        tokens_str=travis_raw.get("tokens_str"),
        tokens=_split_tokens(travis_raw.get("tokens_str")),
    )

    # Date
    date_raw = raw.get("date", {})
    date_cfg = DateCfg(today=date_raw.get("today", "2025-04-02"))

   

    return AppCfg(source=source, path=path_cfg, rq1=rq1_cfg,
                  github=github_cfg, date=date_cfg, travis=travis_cfg, config=config_cfg, raw=raw)