from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"  # e.g., 2019-09-12T17:33:21+02:00


def parse_iso(ts: Optional[str | datetime]) -> Optional[datetime]:
    """Parse ISO-8601 string to timezone-aware datetime (UTC)."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        # If naive datetime (no tz), force UTC
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    s = str(ts).strip()
    if not s or s.lower() == "none":
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            dt = datetime.strptime(s, ISO_FMT)
        except Exception:
            return None
    # If no tz info, assume UTC
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def years_between(start: datetime, end: datetime) -> float:
    """Compute fractional years between two datetimes."""
    return (end - start).total_seconds() / (365.25 * 24 * 3600)


def normalize_lang(lang: Optional[str]) -> Optional[str]:
    """Normalize language string to lowercase."""
    return None if lang is None else str(lang).strip().lower()


def to_bool(x) -> Optional[bool]:
    """Coerce common truthy/falsey strings to bool."""
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def parse_ci_adoption_times(val) -> Dict[str, Optional[str]]:
    """Parse a CI adoption mapping from dict / JSON / python-literal string."""
    if isinstance(val, dict):
        return val
    if val is None:
        return {}
    s = str(val).strip()
    if not s or s.lower() == "none":
        return {}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


@dataclass
class CIAssignment:
    """Resolved CI choice and its adoption time."""
    ci_type: Optional[str]                 # 'GitHubActions' or 'TravisCI'
    ci_adoption_time: Optional[datetime]


def pick_ci(ci_map: Dict[str, Optional[str]]) -> CIAssignment:
    """
    Select CI type/time from the map:
      - consider only GitHubActions and TravisCI
      - if both exist, pick the earliest adoption time
    """
    candidates: List[Tuple[str, datetime]] = []
    for key in ("GitHubActions", "TravisCI"):
        dt = parse_iso(ci_map.get(key))
        if dt:
            candidates.append((key, dt))
    if not candidates:
        return CIAssignment(None, None)
    ci_type, ci_time = sorted(candidates, key=lambda kv: kv[1])[0]
    return CIAssignment(ci_type, ci_time)