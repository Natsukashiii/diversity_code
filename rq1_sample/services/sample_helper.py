# sample_helper.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from config.helper import load_csv, safe_to_csv


# ------------------------- bucket construction -------------------------
@dataclass
class BinSpec:
    """Numeric bin specification."""
    edges: List[float]
    labels: List[str]  # e.g., ["low", "high"] or ["low", "mid", "high"]




def build_sampling_config(
    df: pd.DataFrame,
    sample_fields: List[str],
    *,
    mode: Literal["percentage", "numeric", "both"] = "both",
    numeric_bins: int = 2,
    per_bin_ranges: Optional[List[Tuple[float, float]]] = None,
    low_bin_range_fallback: Tuple[float, float] = (0.0, 0.40),
    high_bin_range_fallback: Tuple[float, float] = (0.60, 1.0),
) -> Dict[str, Optional[Dict[str, dict]]]:
    """
    Build per-field sampling config with 'numeric' bins first,
    then 'percentage' as subsets *inside each bin*.

    Returns a dict:
      {
        "numeric":     Dict[str, dict] | None,   # numeric binning config
        "percentage":  Dict[str, dict] | None    # per-bin percentage config (subset of numeric)
      }
    """

    def _labels_for_bins(k: int) -> List[str]:
        if k == 2:
            return ["low", "high"]
        if k == 3:
            return ["low", "mid", "high"]
        return [f"q{i+1}" for i in range(k)]

    def _quantile_edges(s: pd.Series, k: int) -> List[float]:
        q = np.linspace(0, 1, k + 1)
        edges = list(np.unique(s.dropna().quantile(q).values.astype(float)))
        if len(edges) < 2:
            edges = [float(s.min()), float(s.max())]
        return edges

    # ---- validate per_bin_ranges ----
    if per_bin_ranges is not None:
        if len(per_bin_ranges) != numeric_bins:
            raise ValueError(f"per_bin_ranges length ({len(per_bin_ranges)}) must equal numeric_bins ({numeric_bins}).")
        for rng in per_bin_ranges:
            a, b = rng
            if not (0.0 <= a < b <= 1.0):
                raise ValueError(f"Invalid per-bin range {rng}; must satisfy 0.0 <= a < b <= 1.0.")

    cfg_numeric: Dict[str, dict] = {}
    cfg_percentage: Dict[str, dict] = {}

    for f in sample_fields:
        s = df[f].dropna()
        if s.empty:
            continue

        if pd.api.types.is_numeric_dtype(s):
            # 1) numeric bins
            edges = _quantile_edges(s, numeric_bins)
            if len(edges) < numeric_bins + 1:
                vmin, vmax = float(s.min()), float(s.max())
                edges = [vmin] + [e for e in edges if e not in (vmin, vmax)] + [vmax]
                edges = sorted(set(edges))
                if len(edges) < numeric_bins + 1:
                    edges = [vmin, vmax]
                    numeric_bins = 1
            labels = _labels_for_bins(numeric_bins) if numeric_bins >= 2 else ["q1"]

            numeric_buckets = {}
            for i, lab in enumerate(labels):
                lo = float(edges[i])
                hi = float(edges[i + 1]) if i + 1 < len(edges) else float(edges[-1])
                numeric_buckets[lab] = {"min": lo, "max": hi}

            if mode in ("numeric", "both"):
                cfg_numeric[f] = {
                    "type": "numeric",
                    "labels": labels,
                    "edges": edges,
                    "buckets": numeric_buckets,
                }

            # 2) percentage inside each numeric bin
            percentage_buckets = {}
            per_bin = per_bin_ranges
            if per_bin is None and numeric_bins == 2:
                per_bin = [low_bin_range_fallback, high_bin_range_fallback]

            for i, lab in enumerate(labels):
                lo = float(edges[i])
                hi = float(edges[i + 1]) if i + 1 < len(edges) else float(edges[-1])

                if per_bin is not None:
                    a, b = per_bin[i]
                    s_in_bin = s[(s >= lo) & (s <= hi)]
                    if not s_in_bin.empty:
                        inner_lo = float(s_in_bin.quantile(a))
                        inner_hi = float(s_in_bin.quantile(b))
                        if np.isfinite(inner_lo) and np.isfinite(inner_hi) and inner_lo < inner_hi:
                            percentage_buckets[lab] = {"min": inner_lo, "max": inner_hi}

            if percentage_buckets and mode in ("percentage", "both"):
                cfg_percentage[f] = {
                    "type": "numeric",
                    "labels": labels,
                    "edges": edges,
                    "buckets": percentage_buckets,
                }

        else:
            # categorical passthrough
            values = sorted([str(v) for v in s.astype(str).unique().tolist()])
            if mode in ("numeric", "both"):
                cfg_numeric[f] = {"type": "categorical", "values": values}
            if mode in ("percentage", "both"):
                cfg_percentage[f] = {"type": "categorical", "values": values}

    return {
        "numeric": cfg_numeric if mode in ("numeric", "both") else None,
        "percentage": cfg_percentage if mode in ("percentage", "both") else None,
    }

def build_sampling_config_value(
    df: pd.DataFrame,
    sample_fields: List[str],
    *,
    mode: Literal["percentage", "numeric", "both"] = "both",
    numeric_bins: int = 2,
    per_bin_ranges: Optional[List[Tuple[float, float]]] = None,
    low_bin_range_fallback: Tuple[float, float] = (0.10, 0.45),
    high_bin_range_fallback: Tuple[float, float] = (0.60, 1),
) -> Dict[str, Optional[Dict[str, dict]]]:
    """
    Build sampling configs:
      - 'numeric': value-based equal-width bins over [P1, P99] of the data
                   (use min/max fallback if needed)
      - 'percentage': inside each numeric bin, take a *value-range* sub-interval
        defined by (a,b) in [0,1], i.e. inner_lo = lo + a*(hi-lo)

    Returns
    -------
    {
      "numeric":     Dict[str, dict] | None,
      "percentage":  Dict[str, dict] | None
    }
    """

    def _labels_for_bins(k: int) -> List[str]:
        if k == 2:
            return ["low", "high"]
        if k == 3:
            return ["low", "mid", "high"]
        return [f"q{i+1}" for i in range(k)]

    # ---- validate per_bin_ranges (value-mode needs 0 <= a < b <= 1) ----
    if per_bin_ranges is not None:
        if len(per_bin_ranges) != numeric_bins:
            raise ValueError(
                f"per_bin_ranges length ({len(per_bin_ranges)}) must equal numeric_bins ({numeric_bins})."
            )
        for rng in per_bin_ranges:
            a, b = rng
            if not (0.0 <= a < b <= 1.0):
                raise ValueError(f"Invalid per-bin range {rng}; must satisfy 0.0 <= a < b <= 1.0.")

    cfg_numeric: Dict[str, dict] = {}
    cfg_percentage: Dict[str, dict] = {}

    for f in sample_fields:
        s = df[f].dropna()
        if s.empty:
            continue

        if pd.api.types.is_numeric_dtype(s):
            # -------- 1) numeric bins (value-based equal-width over [P1, P99]) --------
            try:
                p1 = float(np.nanpercentile(pd.to_numeric(s, errors="coerce"), 1))
                p99 = float(np.nanpercentile(pd.to_numeric(s, errors="coerce"), 99))
            except Exception:
                p1, p99 = float(s.min()), float(s.max())

            # fallback if degenerate/not finite
            if not np.isfinite(p1) or not np.isfinite(p99):
                p1, p99 = float(s.min()), float(s.max())

            # if still degenerate, force single bin
            if p1 == p99:
                edges = [p1, p99]
                eff_bins = 1
            else:
                edges = list(np.linspace(p1, p99, max(2, numeric_bins + 1)))
                eff_bins = numeric_bins

            # ensure strictly increasing unique edges; fallback if collapsed
            edges = sorted(set(float(e) for e in edges))
            if len(edges) < 2:
                vmin, vmax = float(s.min()), float(s.max())
                edges = [vmin, vmax]
                eff_bins = 1
            if len(edges) < eff_bins + 1:
                # pad with end points if numerical collapse happened
                while len(edges) < eff_bins + 1:
                    edges.append(edges[-1])

            labels = _labels_for_bins(eff_bins) if eff_bins >= 2 else ["q1"]

            numeric_buckets = {}
            for i, lab in enumerate(labels):
                lo = float(edges[i])
                hi = float(edges[i + 1]) if i + 1 < len(edges) else float(edges[-1])
                numeric_buckets[lab] = {"min": lo, "max": hi}

            if mode in ("numeric", "both"):
                cfg_numeric[f] = {
                    "type": "numeric",
                    "labels": labels,
                    "edges": edges,
                    "buckets": numeric_buckets,
                }

            # -------- 2) percentage (value-based sub-interval inside each bin) --------
            percentage_buckets = {}
            per_bin = per_bin_ranges
            if per_bin is None and len(labels) == 2:
                # default only for 2 bins
                per_bin = [low_bin_range_fallback, high_bin_range_fallback]

            for i, lab in enumerate(labels):
                lo = float(edges[i])
                hi = float(edges[i + 1]) if i + 1 < len(edges) else float(edges[-1])

                if per_bin is None:
                    # no per-bin spec -> skip
                    continue

                a, b = per_bin[i]
                inner_lo = lo + a * (hi - lo)
                inner_hi = lo + b * (hi - lo)

                if not np.isfinite(inner_lo) or not np.isfinite(inner_hi):
                    continue
                if inner_lo >= inner_hi:
                    continue

                percentage_buckets[lab] = {"min": float(inner_lo), "max": float(inner_hi)}

            if percentage_buckets and mode in ("percentage", "both"):
                cfg_percentage[f] = {
                    "type": "numeric",
                    "labels": labels,
                    "edges": edges,   # same outer edges for reference
                    "buckets": percentage_buckets,
                }

        else:
            # categorical passthrough
            values = sorted([str(v) for v in s.astype(str).unique().tolist()])
            if mode in ("numeric", "both"):
                cfg_numeric[f] = {"type": "categorical", "values": values}
            if mode in ("percentage", "both"):
                cfg_percentage[f] = {"type": "categorical", "values": values}

    return {
        "numeric": cfg_numeric if mode in ("numeric", "both") else None,
        "percentage": cfg_percentage if mode in ("percentage", "both") else None,
    }

def apply_buckets(
    df: pd.DataFrame,
    sampling_configs: Union[
        Dict[str, dict],
        List[Dict[str, dict]],
        List[Tuple[str, Dict[str, dict]]]
    ],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Apply one/more sampling configs and return:
      - dict[name -> bucketed_df] with <field>_bucket + bucket_id + bucket_desc + bucket_name
      - bucket_info: one row per bucket_id with metadata and bucket_name
    """
    # --- normalize input to List[(name, config)] ---
    named_cfgs: List[Tuple[str, Dict[str, dict]]] = []
    if isinstance(sampling_configs, dict):
        looks_like_single = all(
            isinstance(v, dict) and any(k in v for k in ("type", "values", "buckets"))
            for v in sampling_configs.values()
        )
        named_cfgs = [("cfg", sampling_configs)] if looks_like_single else list(sampling_configs.items())
    elif isinstance(sampling_configs, list):
        if sampling_configs and isinstance(sampling_configs[0], tuple):
            named_cfgs = [(str(n), c) for (n, c) in sampling_configs]
        else:
            named_cfgs = [(f"cfg{i+1}", c) for i, c in enumerate(sampling_configs)]
    else:
        raise TypeError("sampling_configs must be dict, list[dict], list[(name, dict)], or dict[name->dict].")

    if not named_cfgs:
        return {}, pd.DataFrame(columns=["bucket_id", "bucket_desc", "bucket_name"])

    # enforce same field order across configs
    field_order = list(named_cfgs[0][1].keys())
    for name, cfg in named_cfgs[1:]:
        if list(cfg.keys()) != field_order:
            raise ValueError("All configs must use the same fields and order.")

    # --- helpers ---
    def _assign_label_from_buckets(val: float, buckets: Dict[str, Dict[str, float]]) -> Optional[str]:
        for lab, rng in buckets.items():
            if rng["min"] <= val <= rng["max"]:
                return lab
        return None

    def _key_series(frame: pd.DataFrame) -> pd.Series:
        return frame[[f"{f}_bucket" for f in field_order]].astype(str).apply(tuple, axis=1)

    # --- build per-config bucketed frames (label columns only) ---
    per_cfg_df: Dict[str, pd.DataFrame] = {}
    for name, cfg in named_cfgs:
        fr = df.copy()
        # numeric coercion
        for f, spec in cfg.items():
            if spec.get("type") == "numeric":
                fr[f] = pd.to_numeric(fr[f], errors="coerce")
        # assign bucket labels
        for f, spec in cfg.items():
            col = f"{f}_bucket"
            if spec["type"] == "numeric":
                fr[col] = fr[f].apply(lambda v: _assign_label_from_buckets(v, spec["buckets"]))
            else:
                fr[col] = fr[f].astype(str)
            fr[col] = fr[col].astype("category")
        # drop rows that couldn't be bucketed
        need = [f"{f}_bucket" for f in field_order]
        fr = fr.dropna(subset=need).copy()
        per_cfg_df[name] = fr

    # --- validate percentage ⊂ numeric if both exist ---
    config_names = [n for n, _ in named_cfgs]
    if "numeric" in config_names and "percentage" in config_names:
        cfg_n = dict(named_cfgs)["numeric"]
        cfg_p = dict(named_cfgs)["percentage"]
        for f in field_order:
            sn, sp = cfg_n.get(f), cfg_p.get(f)
            if not sp or sp["type"] != "numeric":
                continue
            if not sn or sn["type"] != "numeric":
                raise ValueError(f"Field '{f}' is numeric in 'percentage' but not in 'numeric'.")
            for lab_p, rng_p in sp["buckets"].items():
                p_lo, p_hi = rng_p["min"], rng_p["max"]
                ok = False
                if lab_p in sn["buckets"]:
                    n_lo, n_hi = sn["buckets"][lab_p]["min"], sn["buckets"][lab_p]["max"]
                    ok = (n_lo <= p_lo) and (p_hi <= n_hi)
                else:
                    for rng_n in sn["buckets"].values():
                        if rng_n["min"] <= p_lo and p_hi <= rng_n["max"]:
                            ok = True; break
                if not ok:
                    raise ValueError(f"percentage[{f}:{lab_p}] range [{p_lo},{p_hi}] not inside any numeric bin.")

    # --- build global key universe and id/desc mapping ---
    all_keys = []
    for name in config_names:
        all_keys.extend(_key_series(per_cfg_df[name]).unique())
    key_universe = pd.Index(sorted(set(all_keys), key=lambda k: tuple(map(str, k))))
    bucket_id_map = {k: i for i, k in enumerate(key_universe)}

    def _desc_from_key(key: Tuple[str, ...]) -> str:
        return " | ".join(f"{f}={lab}" for f, lab in zip(field_order, key))

    key_desc_map = {k: _desc_from_key(k) for k in key_universe}

    # --- stamp id/desc back to each config df (and later bucket_name) ---
    for name in config_names:
        fr = per_cfg_df[name].copy()
        keys = _key_series(fr)
        fr["bucket_id"] = keys.map(bucket_id_map).astype(int)
        fr["bucket_desc"] = keys.map(key_desc_map)
        per_cfg_df[name] = fr

    # --- build bucket_info (one row per bucket_id) with *_level/value metadata ---
    rows = []
    for key, bid in bucket_id_map.items():
        row = {"bucket_id": int(bid), "bucket_desc": key_desc_map[key]}
        for f, lab in zip(field_order, key):
            # detect type from any config (they must agree)
            any_spec = None
            for _, cfg in named_cfgs:
                if f in cfg:
                    any_spec = cfg[f]; break
            if any_spec and any_spec["type"] == "numeric":
                row[f"{f}_level"] = lab
            else:
                row[f"{f}_value"] = lab
        rows.append(row)

    bucket_info = pd.DataFrame(rows).sort_values("bucket_id").reset_index(drop=True)

    # (translate_desc_to_name expects *_level/*_value fields in bucket_info)
    bucket_info["bucket_name"] = bucket_info.apply(lambda r: translate_desc_to_name(r), axis=1)
    id2name = dict(zip(bucket_info["bucket_id"], bucket_info["bucket_name"]))

    for name in config_names:
        fr = per_cfg_df[name].copy()
        fr["bucket_name"] = fr["bucket_id"].map(id2name)
        per_cfg_df[name] = fr

    # counts per config (optional, kept for overview)
    for name in config_names:
        vc = per_cfg_df[name]["bucket_id"].value_counts()
        bucket_info[f"{name}_count"] = bucket_info["bucket_id"].map(vc).fillna(0).astype(int)

    # column order
    count_cols = [c for c in bucket_info.columns if c.endswith("_count")]
    base_cols = ["bucket_id", "bucket_desc", "bucket_name"]
    other_cols = [c for c in bucket_info.columns if c not in (*base_cols, *count_cols)]
    bucket_info = bucket_info[base_cols + count_cols + other_cols]

    return per_cfg_df, bucket_info


def attach_ranges_to_bucket_info(
    bucket_info: pd.DataFrame,
    cfg_numeric: Dict,
    cfg_percentage: Dict,
    fields: List[str],
) -> pd.DataFrame:
    """
    Map range text back to bucket_info for reporting.
    """
    out = bucket_info.copy()

    def _fmt_rng(rng: Dict[str, float]) -> str:
        return f"[{rng['min']:.6f}, {rng['max']:.6f}]"

    for f in fields:
        # 1) numeric
        col_num = f"{f}_numeric_range"
        if f in cfg_numeric and "buckets" in cfg_numeric[f]:
            mp = {lvl: _fmt_rng(rng) for lvl, rng in cfg_numeric[f]["buckets"].items()}
            out[col_num] = out.get(f"{f}_level", "").map(mp).fillna("")
        else:
            out[col_num] = ""

        # 2) percentage
        col_pct = f"{f}_percentage_range"
        if f in cfg_percentage and "buckets" in cfg_percentage[f]:
            mp = {lvl: _fmt_rng(rng) for lvl, rng in cfg_percentage[f]["buckets"].items()}
            out[col_pct] = out.get(f"{f}_level", "").map(mp).fillna("")
        else:
            out[col_pct] = ""

    return out




# --------------------------- reporting/plots ---------------------------
def print_bucket_overview(df: pd.DataFrame, sample_fields: List[str], title: str = "[All] bucket counts") -> None:
    """Print per-bucket counts and totals."""
    bcols = [f"{f}_bucket" for f in sample_fields]
    if not all(c in df.columns for c in bcols):
        print("[WARN] Bucket columns missing; skip overview.")
        return

    print(f"\n{title}")
    counts = df.groupby(bcols, dropna=False).size().reset_index(name="count")
    print(counts.to_string(index=False))


def print_quota_gaps(gaps: pd.DataFrame) -> None:
    """Print quota shortfalls by bucket."""
    if gaps is None or gaps.empty:
        print("\n[Quota] No gaps.")
        return
    missing = gaps[gaps["shortfall"] > 0]
    print("\n[Quota] Gaps (shortfall > 0):")
    if missing.empty:
        print("None.")
    else:
        print(missing.to_string(index=False))



# ------------------------- visualization -------------------------


def visualize(
    dfs: List[pd.DataFrame],
    labels: List[str],
    columns: List[str],
    outpath: Path,
    ncols: int = 2,
    figsize_per_plot: Tuple[float, float] = (3, 2),
    dpi: int = 150,
    clip_percentiles: Tuple[float, float] = (1, 99),
    remove_outliers: bool = True,
    outlier_method: Literal["trim", "winsorize"] = "trim",
) -> None:
    """
    One-page comparison.
    Numeric panel = ECDF(left) + Violin(right).
    Categorical = stacked bars.

    Readability tweaks:
    - ECDF x ticks/label on TOP only; remove bottom ticks; remove y label.
    - Violin y axis on RIGHT.
    - CRITICAL: parent subplot axis is fully disabled for numeric panels
      to avoid stray bottom ticks like 0.2/0.4 from the parent axis.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    base_palette = sns.color_palette("Set2", len(labels))
    label_palette = dict(zip(labels, base_palette))

    nrows = int(np.ceil(len(columns) / ncols)) if columns else 1
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False
    )
    axes = axes.ravel()

    def _pooled_numeric(col: str) -> np.ndarray:
        pooled = []
        for d in dfs:
            if col in d.columns:
                vals = pd.to_numeric(d[col], errors="coerce").dropna().values
                if vals.size:
                    pooled.append(vals)
        return np.concatenate(pooled) if pooled else np.array([])

    def _is_numeric(col: str) -> bool:
        has_any = any(col in d.columns for d in dfs)
        return has_any and all(
            (col not in d.columns) or pd.api.types.is_numeric_dtype(d[col]) for d in dfs
        )

    def _winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").clip(lower=lo, upper=hi)

    def _trim(s: pd.Series, lo: float, hi: float) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s[(s >= lo) & (s <= hi)]

    def _kill_parent_axis(ax: plt.Axes) -> None:
        """Hard-disable parent subplot axis to prevent bottom ticks."""
        ax.set_axis_off()                           # turn off axis artist
        ax.patch.set_visible(False)                 # hide face
        ax.set_xticks([]); ax.set_yticks([])        # extra safety
        for sp in ax.spines.values():               # hide spines
            sp.set_visible(False)

    for ax, col in zip(axes, columns):
        if _is_numeric(col):
            # ---- disable parent axis (fixes lingering bottom ticks) ----
            _kill_parent_axis(ax)

            # global clip range
            lo = hi = None
            pooled = _pooled_numeric(col)
            if clip_percentiles is not None and pooled.size:
                lo, hi = np.nanpercentile(pooled, clip_percentiles)
                if lo == hi:
                    lo, hi = np.min(pooled), np.max(pooled)

            # ECDF (left)
            ax_left = ax.inset_axes([0.00, 0.00, 0.52, 1.00])
            any_plotted = False
            for lab, d in zip(labels, dfs):
                if col not in d.columns:
                    continue
                s = d[col]
                if lo is not None and hi is not None and remove_outliers:
                    s = _trim(s, lo, hi) if outlier_method == "trim" else _winsorize(s, lo, hi)
                else:
                    s = pd.to_numeric(s, errors="coerce")
                s = s.dropna()
                if s.empty:
                    continue
                sns.ecdfplot(x=s.values, ax=ax_left, label=lab,
                             color=label_palette[lab], linewidth=2)

                #  draw "median" vertical dashed line (same color, semi-transparent)
                median_val = np.median(s.values)
                ax_left.axvline(median_val, linestyle="--", linewidth=1,
                                color=label_palette[lab], alpha=0.75)

                any_plotted = True

            # top-only ticks/label; bottom fully removed
            ax_left.xaxis.set_ticks_position("bottom")
            ax_left.tick_params(axis="x", labelbottom=True)

            ax_left.xaxis.set_label_position("top")
            ax_left.set_xlabel(col, labelpad=4)
            ax_left.set_ylabel("")  # remove F(x)
            ax_left.tick_params(axis="x",
                                which="both",
                                top=False, labeltop=False,
                                bottom=True, labelbottom=True)
            ax_left.grid(True, ls="--", alpha=0.3)
            if not remove_outliers and lo is not None and hi is not None:
                ax_left.set_xlim(lo, hi)
            if any_plotted:
                ax_left.legend(fontsize=7, loc="lower right", frameon=False)

            # Violin (right)
            ax_right = ax.inset_axes([0.56, 0.00, 0.44, 1.00])
            plot_df = []
            for lab, d in zip(labels, dfs):
                if col not in d.columns:
                    continue
                s = d[col]
                if lo is not None and hi is not None and remove_outliers:
                    s = _trim(s, lo, hi) if outlier_method == "trim" else _winsorize(s, lo, hi)
                else:
                    s = pd.to_numeric(s, errors="coerce")
                s = s.dropna()
                if s.empty:
                    continue
                plot_df.append(pd.DataFrame({col: s.values, "label": lab}))

            if plot_df:
                plot_df = pd.concat(plot_df, ignore_index=True)
                sns.violinplot(
                    data=plot_df, x="label", y=col,
                    hue="label", palette=label_palette, legend=False,
                    inner=None, linewidth=0, alpha=0.25, cut=0, ax=ax_right, zorder=0
                )
                if not remove_outliers and lo is not None and hi is not None:
                    ax_right.set_ylim(lo, hi)
                ax_right.set_xlabel("")
                ax_right.yaxis.tick_right()
                ax_right.yaxis.set_label_position("right")
                ax_right.set_ylabel("")
                ax_right.grid(True, ls="--", alpha=0.3)
                ax_right.tick_params(axis="x", labelrotation=20, labelsize=8)
            else:
                ax_right.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_right.transAxes)
                ax_right.set_xticks([]); ax_right.set_yticks([])

            for subax in (ax_left, ax_right):
                subax.margins(x=0.02, y=0.05)

        else:
            # categorical: stacked bars
            all_cats: List[str] = []
            for d in dfs:
                if col in d.columns:
                    for c in d[col].dropna().astype(str).str.strip().unique().tolist():
                        if c not in all_cats:
                            all_cats.append(c)

            if not all_cats:
                ax.text(0.5, 0.5, f"{col}: N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                positions = np.arange(len(dfs))
                bottoms = np.zeros(len(dfs), dtype=float)
                for cat in all_cats:
                    heights = []
                    for d in dfs:
                        if col in d.columns:
                            heights.append(int((d[col].astype(str).str.strip() == cat).sum()))
                        else:
                            heights.append(0)
                    ax.bar(positions, heights, bottom=bottoms, label=cat, alpha=0.85)
                    bottoms += np.array(heights, dtype=float)

                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=0)
                ax.set_ylabel("count")
                ax.set_title(f"{col} • Counts", fontsize=9, pad=6)
                ax.legend(title=col, fontsize=6, loc="best", frameon=False)

    # hide unused cells
    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.tight_layout(pad=1.2)
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved visualization → {outpath}")



def translate_desc_to_name(row: dict) -> str:
    """
    Translate bucket description into a 4-letter name code.
    Rules:
      1st letter: language (J=Java, P=Python)
      2nd letter: CI type (T=TravisCI, G=GitHubActions)
      3rd letter: project_age_days (A=high, a=low)
      4th letter: avg_issues_close_time (I=high, i=low)
    """
    # 1. language
    lang = row.get("mainLanguage_value", "")
    if lang == "Java":
        c1 = "J"
    elif lang == "Python":
        c1 = "P"
    else:
        c1 = "X"  # fallback
    
    # 2. ci_type
    ci = row.get("ci_type_value", "")
    if ci == "TravisCI":
        c2 = "T"
    elif ci == "GitHubActions":
        c2 = "G"
    else:
        c2 = "X"
    
    # 3. project_age_days bucket
    age = row.get("project_age_days_level", "").lower()
    if age == "high":
        c3 = "A"
    elif age == "low":
        c3 = "a"
    else:
        c3 = "x"
    
    # 4. avg_issues_close_time bucket
    issue_time = row.get("open_issue_ratio_level", "").lower()
    if issue_time == "high":
        c4 = "I"
    elif issue_time == "low":
        c4 = "i"
    else:
        c4 = "x"

    return f"{c1}{c2}{c3}{c4}"





# ---------- simple dimension-level counts ----------
DIMENSION_SETS = [
    ["mainLanguage"],
    ["ci_type"],
    ["project_age_days_bucket"],
    ["open_issue_ratio_bucket"],
]

def simple_counts(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Return counts grouped only by the given columns (no full context).
    Adds a 'repos' column with counts and a 'Total' row.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}. Will skip.")
        return pd.DataFrame(columns=cols + ["repos"])

    g = (df
         .groupby(cols, dropna=False)
         .size()
         .reset_index(name="repos")
         .sort_values("repos", ascending=False)
         .reset_index(drop=True))
    total = pd.DataFrame({c: ["ALL"] for c in cols})
    total["repos"] = [int(g["repos"].sum())]
    g = pd.concat([g, total], ignore_index=True)
    return g

def print_and_save_simple_counts(name: str,
                                 df: pd.DataFrame,
                                 out_dir: Path,
                                 dimension_sets: list[list[str]] = DIMENSION_SETS) -> None:
    """
    For each dimension set (e.g., ['mainLanguage']), print and save counts:
    <name>_<dim>_counts.csv
    """
    for dims in dimension_sets:
        tag = "_".join(dims)
        print(f"\n[Counts] {name} by {tag}")
        counts = simple_counts(df, dims)
        print(counts.to_string(index=False))
        # out_path = out_dir / f"{name.lower()}_{tag}_counts.csv"
        # safe_to_csv(counts, out_path)
        # print(f"[INFO] Saved: {out_path}")