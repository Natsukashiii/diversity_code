# rq3_plot.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from config.load_config import load_config

cfg = load_config()
RQ3_DATA_PATH  = Path(str(cfg.path.rq3_data_path))
RQ3_PLOT_PATH  = Path(str(cfg.path.rq3_plot_path))

FILE_BUCKET_STRA   = RQ3_DATA_PATH / "rdd_by_bucket_stra.csv"
FILE_OVERALL_STRA  = RQ3_DATA_PATH / "rdd_overall_stra.csv"
FILE_OVERALL_QUOTA = RQ3_DATA_PATH / "rdd_overall_quota.csv"

METRICS = [
    "commit_frequency","modified_files","code_churn_size",
    "issues_open_count","issues_close_count",
    "prs_open_count","prs_close_count",
    "pr_latency","pr_merge_ratio","commit_merge_ratio","releases_count",
]
DIM_SPECS = [
    ("Language (P/J)", 0, ["P", "J"]),   # 语言：P 在前、J 在后
    ("CI Tool (T/G)",  1, ["T", "G"]),   # CI 工具：T 在前、G 在后
    ("Project Age (A/a)", 2, ["A", "a"]),# 年龄：A(older) 在前、a(younger) 在后
    ("Open Issue Ratio (I/i)", 3, ["I", "i"]), # open issue ratio：I(high) 在前、i(low) 在后
]

def _contexts_for_all_metrics() -> dict[str, dict]:
    """
    返回 {metric: {'contexts': [...], 'n_repos': int, 'n_buckets': int}}
    """
    data = {}
    for m in METRICS:
        ctxs, n_repos, n_buckets = _contexts_and_repo_for_metric(m)
        if ctxs:
            data[m] = {"contexts": ctxs, "n_repos": n_repos, "n_buckets": n_buckets}
    return data


def _context_code_from_group_name(s: str) -> str | None:
    """
    优先直接匹配 4 位上下文码（支持大小写混合），否则再尝试关键词解析。
    """
    if not isinstance(s, str) or not s:
        return None

    # 1) 直接匹配 4 字码：任意位置出现 JP × TG × Aa × Ii
    m = re.search(r'([JP][TG][Aa][Ii])', s)
    if m:
        return m.group(1)

    low = s.lower()

    # 2) 关键词解析（更宽松）
    lang = None
    if "java" in low or re.search(r'\bj\b', low):
        lang = "J"
    elif "python" in low or "py" in low or re.search(r'\bp\b', low):
        lang = "P"

    ci = None
    if "travis" in low or re.search(r'\bt\b', low):
        ci = "T"
    elif "github actions" in low or "githubactions" in low or "gha" in low or re.search(r'\bg\b', low):
        ci = "G"

    age = None
    if "older" in low or "old" in low:
        age = "A"
    elif "younger" in low or "young" in low:
        age = "a"

    issue = None
    if "high" in low or "higher" in low or re.search(r'\bi\b(?=[^a-z]|$)', low):  # 结尾单独的 i
        issue = "I"
    elif "low" in low or "lower" in low:
        issue = "i"

    if None in (lang, ci, age, issue):
        return None
    return f"{lang}{ci}{age}{issue}"

def _contexts_and_repo_for_metric(metric: str) -> tuple[list[str], int, int]:
    """
    返回 (contexts, repo_total, n_buckets)，其中：
      - contexts: 该 metric 下所有“与 STRA overall 相反”的 bucket 的 context 4位码列表
      - repo_total: 这些“反向” bucket 的 n_projects 之和
      - n_buckets: 这些“反向” bucket 的数量
    """
    b = _read_csv(FILE_BUCKET_STRA)
    os = _read_csv(FILE_OVERALL_STRA)

    b = b[b["metric"] == metric].copy()
    os = os[os["metric"] == metric].copy()

    if os.empty or b.empty:
        return [], 0, 0

    # 数值化
    b["coef_slope_delta"] = pd.to_numeric(b["coef_slope_delta"], errors="coerce")
    b["n_projects"] = pd.to_numeric(b.get("n_projects", 0), errors="coerce").fillna(0).astype(int)

    stra_delta = pd.to_numeric(os["coef_slope_delta"].iloc[0], errors="coerce")
    if not pd.notna(stra_delta) or float(stra_delta) == 0.0:
        return [], 0, 0

    stra_sign = np.sign(float(stra_delta))
    b = b[pd.notna(b["coef_slope_delta"])].copy()
    if b.empty:
        return [], 0, 0

    b["opp"] = np.sign(b["coef_slope_delta"].astype(float)) * stra_sign < 0
    opp = b[b["opp"]]
    if opp.empty:
        return [], 0, 0

    # contexts
    contexts = []
    for _, r in opp.iterrows():
        code = _context_code_from_group_name(str(r.get("group_name", "")))
        if code is not None:
            contexts.append(code)

    repo_total = int(opp["n_projects"].sum())
    n_buckets = int(len(opp))
    return contexts, repo_total, n_buckets

def plot_opposite_context_dimensions(outdir: Path):

    m2ctx = _contexts_for_all_metrics()
    if not m2ctx:
        print("[WARN] opposite-by-dimension: no data to plot.")
        return

    # —— build counts/perc  —— #
    cols = pd.MultiIndex.from_tuples(
        [(dim[0], lab) for dim in DIM_SPECS for lab in dim[2]],
        names=["dimension", "level"]
    )
    counts = pd.DataFrame(index=METRICS, columns=cols, data=0, dtype=float)
    repo_total = []
    bucket_total = []

    for m in METRICS:
        ctxs = m2ctx.get(m, {}).get("contexts", [])
        n_repos = int(m2ctx.get(m, {}).get("n_repos", 0))
        n_buckets = int(m2ctx.get(m, {}).get("n_buckets", 0)) 
        repo_total.append(n_repos)
        bucket_total.append(n_buckets)                         

        for (dim_title, idx, order) in DIM_SPECS:
            n0 = sum(1 for c in ctxs if len(c) >= 4 and c[idx] == order[0])
            n1 = sum(1 for c in ctxs if len(c) >= 4 and c[idx] == order[1])
            counts.loc[m, (dim_title, order[0])] = n0
            counts.loc[m, (dim_title, order[1])] = n1

    counts["repo_total"] = repo_total
    counts["bucket_total"] = bucket_total     


    # 行内对每个维度做归一化
    perc = counts.copy()
    for (dim_title, _, order) in DIM_SPECS:
        s = counts[(dim_title, order[0])] + counts[(dim_title, order[1])]
        perc[(dim_title, order[0])] = np.where(s > 0, counts[(dim_title, order[0])] / s, 0.0)
        perc[(dim_title, order[1])] = np.where(s > 0, counts[(dim_title, order[1])] / s, 0.0)
    perc["repo_total"]   = repo_total         
    perc["bucket_total"] = bucket_total       

    # —— export CSV —— #
    outdir.mkdir(parents=True, exist_ok=True)
    # counts.to_csv(outdir / "opposite_by_dimension_counts.csv")
    # perc.to_csv(outdir / "opposite_by_dimension_percents.csv", float_format="%.6f")

    # —— plpt —— #
    n_metrics = len(METRICS)
    fig_h = 0.42 * max(4, n_metrics) + 1.1
    fig_w = 10.0
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(fig_w, fig_h), sharey=True, constrained_layout=False)
    plt.subplots_adjust(wspace=0.15, left=0.20, right=0.92, top=0.92, bottom=0.1)

    y = np.arange(n_metrics)[::-1]  
    y_labels = [
    f"{m} (n={m2ctx.get(m, {}).get('n_repos', 0)}; k={m2ctx.get(m, {}).get('n_buckets', 0)})"
    for m in METRICS]

    for ax, (dim_title, _, order) in zip(axes, DIM_SPECS):
        w0 = perc[(dim_title, order[0])].reindex(METRICS).values
        w1 = perc[(dim_title, order[1])].reindex(METRICS).values

        left = np.zeros_like(w0, dtype=float)
        ax.barh(y, w0, left=left, edgecolor="black", linewidth=0.3, label=order[0])
        left += w0
        ax.barh(y, w1, left=left, edgecolor="black", linewidth=0.3, label=order[1])

        ax.set_title(dim_title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v*100)}%"))
        ax.tick_params(axis="x", labelsize=8)

        if ax is axes[0]:
            ax.set_yticks(y)
            ax.set_yticklabels(y_labels, fontsize=8)
        else:
            ax.set_yticks(y)
            ax.tick_params(axis="y", left=False, labelleft=False)

        ax.legend(frameon=True, fontsize=8, loc="lower right")

    fig.suptitle("Opposite-to-Stratified — (n:repo_count k:bucket)", fontsize=11, y=0.98)

    fname = outdir / "opposite_by_dimension"
    fig.savefig(f"{fname}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {fname}.pdf, and CSVs (counts/percents with repo_total).")


def _contexts_by_sign_for_metric(metric: str) -> dict:
    """
    返回:
      {
        'pos': {'contexts': [...], 'n_repos': int, 'n_buckets': int},
        'neg': {'contexts': [...], 'n_repos': int, 'n_buckets': int}
      }
    """
    b = _read_csv(FILE_BUCKET_STRA)
    b = b[b["metric"] == metric].copy()
    if b.empty:
        return {'pos': {'contexts': [], 'n_repos': 0, 'n_buckets': 0},
                'neg': {'contexts': [], 'n_repos': 0, 'n_buckets': 0}}

    b["coef_slope_delta"] = pd.to_numeric(b["coef_slope_delta"], errors="coerce")
    b["n_projects"] = pd.to_numeric(b.get("n_projects", 0), errors="coerce").fillna(0).astype(int)
    b = b[pd.notna(b["coef_slope_delta"])]

    pos = b[b["coef_slope_delta"] > 0]
    neg = b[b["coef_slope_delta"] < 0]

    def _pack(df):
        ctxs = []
        for _, r in df.iterrows():
            c = _context_code_from_group_name(str(r.get("group_name", "")))
            if c:
                ctxs.append(c)
        return {
            "contexts": ctxs,
            "n_repos": int(df["n_projects"].sum()) if not df.empty else 0,
            "n_buckets": int(len(df))
        }

    return {"pos": _pack(pos), "neg": _pack(neg)}


def plot_context_dimensions_by_sign(outdir: Path):
    # —— collect —— #
    metrics_data = {m: _contexts_by_sign_for_metric(m) for m in METRICS}

    cols = pd.MultiIndex.from_tuples(
        [(dim[0], lab, sign) for dim in DIM_SPECS for lab in dim[2] for sign in ("pos", "neg")],
        names=["dimension", "level", "sign"]
    )
    counts = pd.DataFrame(index=METRICS, columns=cols, data=0, dtype=float)


    extra_cols = pd.MultiIndex.from_product([["repo_total", "bucket_total"], ["pos", "neg"]])
    extras = pd.DataFrame(index=METRICS, columns=extra_cols, data=0, dtype=float)

    for m in METRICS:
        for sign in ("pos", "neg"):
            ctxs = metrics_data[m][sign]["contexts"]
            extras.loc[m, ("repo_total", sign)] = metrics_data[m][sign]["n_repos"]
            extras.loc[m, ("bucket_total", sign)] = metrics_data[m][sign]["n_buckets"]
            for (dim_title, idx, order) in DIM_SPECS:
                n0 = sum(1 for c in ctxs if len(c) >= 4 and c[idx] == order[0])
                n1 = sum(1 for c in ctxs if len(c) >= 4 and c[idx] == order[1])
                counts.loc[m, (dim_title, order[0], sign)] = n0
                counts.loc[m, (dim_title, order[1], sign)] = n1

    # 百分比
    perc = counts.copy()
    for (dim_title, _, order) in DIM_SPECS:
        for sign in ("pos", "neg"):
            s = counts[(dim_title, order[0], sign)] + counts[(dim_title, order[1], sign)]
            perc[(dim_title, order[0], sign)] = np.where(s > 0, counts[(dim_title, order[0], sign)] / s, 0.0)
            perc[(dim_title, order[1], sign)] = np.where(s > 0, counts[(dim_title, order[1], sign)] / s, 0.0)

    # —— 导出 CSV —— #
    outdir.mkdir(parents=True, exist_ok=True)
    counts_out = pd.concat([counts, extras], axis=1)
    perc_out = pd.concat([perc, extras], axis=1)
    # counts_out.to_csv(outdir / "by_sign_counts.csv")
    # perc_out.to_csv(outdir / "by_sign_percents.csv", float_format="%.6f")

    # —— 画图（2 行 × 4 列） —— #
    n_metrics = len(METRICS)
    fig_h = 0.44 * max(4, n_metrics) * 2 + 1.2   # 两行
    fig_w = 10.0
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(fig_w, fig_h), sharey=True, constrained_layout=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.25, left=0.20, right=0.92, top=0.92, bottom=0.08)

    y = np.arange(n_metrics)[::-1]

    # y 轴标签显示：metric + 各 sign 的样本量（bucket 数），便于对比
    y_labels_pos = [f"{m} (+k={int(extras.loc[m, ('bucket_total','pos')])})" for m in METRICS]
    y_labels_neg = [f"{m} (-k={int(extras.loc[m, ('bucket_total','neg')])})" for m in METRICS]

    for row, sign, title_suffix, ylabs in [
        (0, "pos", "Δ > 0", y_labels_pos),
        (1, "neg", "Δ < 0", y_labels_neg),
    ]:
        for col, (dim_title, _, order) in enumerate(DIM_SPECS):
            ax = axes[row, col]
            w0 = perc[(dim_title, order[0], sign)].reindex(METRICS).values
            w1 = perc[(dim_title, order[1], sign)].reindex(METRICS).values

            left = np.zeros_like(w0, dtype=float)
            ax.barh(y, w0, left=left, edgecolor="black", linewidth=0.3, label=order[0])
            left += w0
            ax.barh(y, w1, left=left, edgecolor="black", linewidth=0.3, label=order[1])

            ax.set_title(f"{dim_title}\n({title_suffix})", fontsize=10)
            ax.set_xlim(0, 1)
            ax.grid(axis="x", linestyle=":", alpha=0.35)
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v*100)}%"))
            ax.tick_params(axis="x", labelsize=8)

            if col == 0:
                ax.set_yticks(y)
                ax.set_yticklabels(ylabs, fontsize=8)
            else:
                ax.set_yticks(y)
                ax.tick_params(axis="y", left=False, labelleft=False)

            # 每个子图都带上简单图例（两段标签）
            ax.legend(frameon=True, fontsize=8, loc="lower right")

    fig.suptitle("Context distribution - split by adoption slope (Δ>0 vs Δ<0)", fontsize=11, y=0.98)

    fname = outdir / "by_sign_dimension"
    fig.savefig(f"{fname}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {fname}.pdf, and CSVs (by_sign_counts / by_sign_percents).")

# ------------------ PATCH main() to also produce the new figure ------------------

        
def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def _fmt_bucket_name(s: str) -> str:
    if not isinstance(s, str) or not s or s.lower() == "overall":
        return "overall"
    # you can customize to your compact code mapping here if needed
    parts = [x.split("=")[-1].strip() for x in s.split("|")]
    return " ".join([x[:4] for x in parts]) or s

def _collect(metric: str) -> pd.DataFrame:
    b = _read_csv(FILE_BUCKET_STRA)
    os = _read_csv(FILE_OVERALL_STRA)
    oq = _read_csv(FILE_OVERALL_QUOTA)
    b = b[b["metric"] == metric].copy()
    os = os[os["metric"] == metric].copy()
    oq = oq[oq["metric"] == metric].copy()

    rows = []
    if not os.empty:
        rows.append(dict(kind="overall_stra", label="Stratified", is_overall=True,
                         delta=float(os["coef_slope_delta"].iloc[0]),
                         pval=float(os["p_slope_delta"].iloc[0]),
                         n_projects=int(os["n_projects"].iloc[0])))
    if not oq.empty:
        rows.append(dict(kind="overall_quota", label="Quota", is_overall=True,
                         delta=float(oq["coef_slope_delta"].iloc[0]),
                         pval=float(oq["p_slope_delta"].iloc[0]),
                         n_projects=int(oq["n_projects"].iloc[0])))

    if not b.empty:
        b["label"] = b["group_name"].map(_fmt_bucket_name)
        for _, r in b.iterrows():
            rows.append(dict(kind="bucket", label=str(r["label"]), is_overall=False,
                             delta=np.float64(r["coef_slope_delta"]) if pd.notna(r["coef_slope_delta"]) else np.nan,
                             pval=np.float64(r["p_slope_delta"]) if pd.notna(r["p_slope_delta"]) else np.nan,
                             n_projects=int(r["n_projects"]) if pd.notna(r["n_projects"]) else 0))
    df = pd.DataFrame(rows)
    # keep overall first (STRA then QUOTA), then buckets by delta desc
    overall = df[df["is_overall"]]
    buckets = df[~df["is_overall"]].sort_values("delta", ascending=False)
    return pd.concat([overall, buckets], ignore_index=True)

def _z_from_p_two_sided(p):
    # guard tiny p to avoid inf
    p = np.clip(p, 1e-300, 1.0)
    return norm.isf(p/2.0)

@dataclass
class Style:
    w: float = 2.4
    h: float = 4.0
    left_ratio: float = 0.75
    color_overall_stra: str = "#1f77b4"  # blue
    color_overall_quota: str = "#ff7f0e" # orange
    fill_sig: str = "#2ca02c"            # green fill for FDR significant
    edge_sig: str = "#1b7f1b"
    fill_nonsig: str = "white"           # empty square for non-sig
    edge_nonsig: str = "#9e9e9e"
    repo_bar: str = "#bdbdbd"
    zero_line: str = "#555555"

S = Style()

def plot_metric(metric: str, outdir: Path):
    df = _collect(metric)
    if df.empty:
        print(f"[WARN] no data for {metric}")
        return

    # Compute SE and 95% CI from (coef, p) if p available; otherwise CI=NaN
    z = _z_from_p_two_sided(df["pval"].fillna(1.0).values.astype(float))
    coef = df["delta"].values.astype(float)
    se = np.divide(np.abs(coef), z, out=np.full_like(coef, np.nan), where=(z>0))
    ci_lo = coef - 1.96*se
    ci_hi = coef + 1.96*se

    # FDR over buckets only
    mask_bucket = ~df["is_overall"].values
    p_bucket = df.loc[mask_bucket, "pval"].values.astype(float)
    sig_bucket = np.array([False]*len(df))
    if np.isfinite(p_bucket).sum() > 0:
        rej, p_adj, _, _ = multipletests(p_bucket, alpha=0.05, method="fdr_bh")
        sig_bucket[mask_bucket] = rej
    else:
        sig_bucket[:] = False

    # Opposite-to-STRA marker (sign different to STRA overall)
    stra_row = df[df["kind"]=="overall_stra"]
    stra_sign = np.sign(stra_row["delta"].iloc[0]) if not stra_row.empty and pd.notna(stra_row["delta"].iloc[0]) else 0.0
    opposite = (np.sign(coef) * stra_sign < 0) & mask_bucket

    # Y positions (top to bottom)
    y = np.arange(len(df))[::-1]

    # Figure
    fig = plt.figure(figsize=(S.w, S.h))
    gs = fig.add_gridspec(1, 2, width_ratios=[S.left_ratio, 1-S.left_ratio], wspace=0.0001)
    axL = fig.add_subplot(gs[0,0])
    axR = fig.add_subplot(gs[0,1], sharey=axL)

    # Zero line
    axL.axvline(0, color=S.zero_line, lw=1.0, alpha=0.9)

    # ---- plot overall rows: dots + thin CI ----
    for i, row in df[df["is_overall"]].iterrows():
        yi = y[i]
        # color = S.color_overall_stra if row["kind"] == "overall_stra" else S.color_overall_quota
        color = "black"
        if pd.notna(ci_lo[i]) and pd.notna(ci_hi[i]):
            axL.plot([ci_lo[i], ci_hi[i]], [yi, yi], color=color, lw=1.2)
        axL.scatter([coef[i]], [yi], s=36, color=color, zorder=3)



    # ---- plot buckets: same-direction = black circle; opposite = gray triangle ----
    for i, row in df[~df["is_overall"]].iterrows():
        yi = y[i]
        is_opposite = bool(opposite[i])

        # 样式逻辑
        if is_opposite:
            marker = "^"            # 反向 = 三角形
            color = "#7a7a7a"       # 浅灰黑
        else:
            marker = "o"            # 同向 = 圆形
            color = "black"         # 深黑

        # 置信区间线
        if pd.notna(ci_lo[i]) and pd.notna(ci_hi[i]):
            axL.plot([ci_lo[i], ci_hi[i]], [yi, yi], color=color, lw=0.8, alpha=0.9)

        # 中心点
        axL.scatter([coef[i]], [yi],
                    s=50,
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    linewidths=1.0,
                    zorder=3)


    # Y labels/axes
    axL.set_yticks(y)
    axL.set_yticklabels(df["label"].values,fontsize=8)
    axL.set_xlabel("")

    for label in axL.get_yticklabels():
        text = label.get_text()
        if text in ["Stratified", "Quota"]:
            label.set_fontweight("bold")
            label.set_fontsize(9)  

    # ---- right panel: repo counts ----
    mask_bucket = ~df["is_overall"].values
    repos_all = df["n_projects"].fillna(0).astype(int).values
    repos = np.where(mask_bucket, repos_all, 0) 
    # repos = df["n_projects"].fillna(0).astype(int).values
    axR.barh(y, repos, color=S.repo_bar, edgecolor="black", linewidth=0.6)
    axR.set_xlabel("")
    axR.set_yticks(y)
    axR.tick_params(axis="y", left=False, labelleft=False) 
    axR.grid(axis="x", linestyle=":", alpha=0.4)

    axL.spines['right'].set_linewidth(0.8)  
    axL.spines['right'].set_color("#555555")  
    axR.spines['left'].set_visible(False)


    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='black', lw=0, markersize=6, label='Stratified / Quota'),
        Line2D([0],[0], marker='o', color='black', lw=1, markersize=6, markerfacecolor='none', label='Bucket (same trend)'),
        Line2D([0],[0], marker='^', color='#7a7a7a', lw=1, markersize=6, markerfacecolor='none', label='Bucket (opposite trend)'),
    ]
    # axL.legend(handles=legend_elems, loc='lower right', frameon=True, fontsize=7)

    for ax in [axL, axR]:
        ax.tick_params(axis="x", labelsize=9)       
        for label in ax.get_xticklabels():
            label.set_rotation(45)                  
            label.set_ha("right")   

    for ax in [axL, axR]:
        ax.tick_params(axis="x", pad=-1.5)    


    axR.set_xticklabels([]) 
    axR.text(
        0.5, -0.04, "repo_count",
        transform=axR.transAxes,
        ha="center", va="top",
        fontsize=5.5, color="black"
    )             


    # Separator under overall block
    n_overall = int(df["is_overall"].sum())
    if 0 < n_overall < len(df):
        sep_y = len(df) - n_overall - 0.5
        x0, x1 = axL.get_xlim()
        axL.hlines(
            sep_y, x0, x1,
            colors="black",         
            linestyles="dashed",    
            linewidth=0.3,          
            alpha=0.6            
        )
    # Save

    
    x_all = np.concatenate([ci_lo, ci_hi])
    x_all = x_all[np.isfinite(x_all)]
    if len(x_all) > 0:
        x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
        x_abs_max = np.nanmax(np.abs([x_min, x_max]))
        axL.set_xlim(-0.8 * x_abs_max, 0.8 * x_abs_max) 


    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"{metric}"

    fig.subplots_adjust(
    left=0.35,  
    right=0.98,
    top=0.98,
    bottom=0.10, 
    wspace=0.02
    )
    fig.savefig(f"{fname}.pdf", dpi=300, bbox_inches=None)


    plt.close(fig)

def save_horizontal_legend(out_path: Path):
    """Generate a standalone horizontal legend figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elems = [
        # Overall points
        Line2D([0], [0], marker='o', color='black', lw=0, markersize=6,
            label='Stratified / Quota (point)'),
        # Buckets: same vs opposite to STRA
        Line2D([0], [0], marker='o', color='black', lw=1, markersize=6,
            markerfacecolor='none', label='Bucket (same trend with Straified)'),
        Line2D([0], [0], marker='^', color='#7a7a7a', lw=1, markersize=6,
            markerfacecolor='none', label='Bucket (opposite trend)'),
        # 95% CI line（横线）
        Line2D([0, 1], [0, 0], color='black', lw=1.0, label='95% conf. interval'),
        # 右侧灰条（repo 数量）
        Patch(facecolor=S.repo_bar, edgecolor='black', label='Repo count (right)'),
    ]

    fig, ax = plt.subplots(figsize=(1.2, 0.1)) 
    ax.axis('off')

    legend = ax.legend(
        handles=legend_elems,
        loc='center',
        frameon=False,
        ncol=5,                  
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.5
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] legend saved to {out_path}")
# ------------------ batch runner ------------------

def main():
    outdir = Path(RQ3_PLOT_PATH) / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    for m in METRICS:
        try:
            plot_metric(m, outdir)
            print(f"[OK] saved: {m}")
        except Exception as e:
            print(f"[ERR] {m}: {e}")
    save_horizontal_legend(outdir / "legend.pdf")

    plot_opposite_context_dimensions(outdir) 
    plot_context_dimensions_by_sign(outdir)  


if __name__ == "__main__":
    main()