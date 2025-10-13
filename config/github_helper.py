import os
import random
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter, Retry

# ---------- config tokens ----------
# def _load_tokens_from_cfg_or_env(cfg_tokens: Optional[List[str]] = None) -> List[str]:
#     """
#     Priority:
#       1) explicit list (cfg.github.tokens)
#       2) env: GITHUB_TOKENS (comma-separated)
#     """
#     if cfg_tokens:
#         return [t.strip() for t in cfg_tokens if t and t.strip()]
#     cfg = load_config()
#     tokens = cfg.github.tokens or []
#     return tokens

# ---------- session pool ----------
def _make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=False,  # retry any
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_SESSION = _make_session()

# ---------- core request with token rotation ----------

def request_github_tokens_pool(
    tokens: Optional[List[str]],
    url: str,
    *,
    method: str = "GET",
    params: Optional[Dict] = None,
    json: Optional[Dict] = None,
    data: Optional[Union[Dict, str, bytes]] = None,   # <-- add
    timeout: int = 30,
    max_retries_per_token: int = 2,
    headers: Optional[Dict[str, str]] = None,         # optional extra headers
    **kwargs,                                         # <-- pass-through (e.g., stream, auth)
) -> requests.Response:
    """
    GitHub REST call with:
      - token rotation
      - retries + basic rate-limit handling
      - supports params/json/data and extra **kwargs
    """
    if not tokens:
        raise RuntimeError("No GitHub tokens provided")

    if not url.startswith("http"):
        url = urljoin("https://api.github.com/", url.lstrip("/"))

    last_err = None
    for token in random.sample(tokens, k=len(tokens)):
        base_headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "rq-suite/1.0",
        }
        merged_headers = {**base_headers, **(headers or {})}

        for attempt in range(1, max_retries_per_token + 1):
            try:
                resp = _SESSION.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    data=data,                # <-- now supported
                    headers=merged_headers,
                    timeout=timeout,
                    **kwargs,                 # <-- forward extras
                )

                # primary rate limit: exhausted
                if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                    break  # try next token

                # secondary/abuse limits
                if resp.status_code in (403, 429) and "rate limit" in (resp.text or "").lower():
                    time.sleep(2.0)
                    break

                # be gentle with Search API
                if "/search/" in urlparse(url).path:
                    time.sleep(2.0)

                resp.raise_for_status()
                return resp

            except requests.RequestException as e:
                last_err = e
                if attempt < max_retries_per_token:
                    time.sleep(0.8 * attempt)
                # else: fall through to next token

    if last_err:
        raise last_err
    raise RuntimeError("All tokens failed")

# ---------- pagination helper ----------
def _parse_next_link(link_header: Optional[str]) -> Optional[str]:
    """Parse GitHub 'Link' header to find next page URL."""
    if not link_header:
        return None
    parts = [p.strip() for p in link_header.split(",")]
    for p in parts:
        if 'rel="next"' in p:
            seg = p.split(";")[0].strip()
            if seg.startswith("<") and seg.endswith(">"):
                return seg[1:-1]
    return None

def iter_github_items(
    url: str,
    *,
    params: Optional[Dict] = None,
    tokens: Optional[List[str]] = None,
    per_page: int = 100,
    max_pages: Optional[int] = None,
    stop_after: Optional[int] = None,
) -> Iterator[dict]:
    """
    Iterate over all items across pages for a list-style REST v3 endpoint.

    Args:
      url: endpoint path or full url (e.g. '/repos/owner/repo/issues')
      params: query params; 'per_page' & 'page' managed internally
      tokens: list of tokens (defaults to cfg.github.tokens or env)
      per_page: 1..100
      max_pages: hard cap on pages to fetch
      stop_after: stop after yielding N items

    Yields:
      dict items from each page (the JSON array elements)
    """
    if not url.startswith("http"):
        url = urljoin("https://api.github.com/", url.lstrip("/"))

    q = dict(params or {})
    q["per_page"] = max(1, min(100, per_page))
    page = int(q.get("page", 1))
    yielded = 0
    pages_fetched = 0

    while True:
        q["page"] = page
        resp = request_github_tokens_pool(tokens, url, params=q, )
        data = resp.json()

        # List endpoints return JSON array; search returns object with 'items'
        if isinstance(data, dict) and "items" in data:
            items = data.get("items", [])
        elif isinstance(data, list):
            items = data
        else:
            # Not a list-like response
            return

        for item in items:
            yield item
            yielded += 1
            if stop_after and yielded >= stop_after:
                return

        pages_fetched += 1
        if max_pages and pages_fetched >= max_pages:
            return

        next_url = _parse_next_link(resp.headers.get("Link"))
        if not next_url:
            return
        url = next_url  # next page absolute URL
        page += 1


# from config.github_helper import request_github


# resp = request_github("GET", "https://api.github.com/search/repositories",
#                       params={"q": "language:python stars:>1000", "per_page": 10},
#                       cfg=cfg)   # defaults to cfg.github.tokens
# data = resp.json()


# for c in iter_github_items("/repos/pallets/flask/commits", stop_after=500):
#     print(c["sha"], c["commit"]["author"]["date"])