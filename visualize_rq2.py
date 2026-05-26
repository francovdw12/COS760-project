"""visualize_rq2.py

Unified RQ2 visualiser. Generates seven figures:

  Three-panel figures (one per alignment method):
    rq2_cca_panels.png
    rq2_kcca_panels.png
    rq2_vecmap_panels.png
      Panel 1 — F1 learning curve (line per language)
      Panel 2 — CKA heatmap (languages x fractions)
      Panel 3 — BLI p@5 heatmap (languages x fractions)

  Summary figures:
    rq2_learning_curves.png
    rq2_breakeven_table.png
    rq2_conjunctive_vs_disjunctive.png
    rq2_method_heatmap.png

Usage:
    python visualize_rq2.py --results-csv results/rq2/full_baseline_v2/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BREAKEVEN_F1 = 0.03

LANGUAGE_ORDER = ["tsn", "nso", "zul"]

LANGUAGE_LABELS = {
    "tsn": "Setswana\n(disjunctive)",
    "nso": "Sepedi\n(disjunctive)",
    "zul": "isiZulu\n(conjunctive)",
}

LANGUAGE_LABELS_FLAT = {
    "tsn": "Setswana",
    "nso": "Sepedi",
    "zul": "isiZulu",
}

LANG_DISPLAY = {"zul": "isiZulu", "nso": "Sepedi", "tsn": "Setswana"}
LANG_ORDER_DISPLAY = ["isiZulu", "Sepedi", "Setswana"]

LANGUAGE_COLORS = {
    "tsn": "#2A9D8F",
    "nso": "#E9C46A",
    "zul": "#E63946",
}

METHOD_ORDER = ["CCA", "KCCA", "VecMap"]
METHOD_PALETTE = {
    "CCA":    "#4C72B0",
    "KCCA":   "#C44E52",
    "VecMap": "#55A868",
}

MORPHO_PALETTE = {
    "Conjunctive (isiZulu)":           "#E07B54",
    "Disjunctive (Sepedi + Setswana)": "#5BA4CF",
}

FRACTION_ORDER  = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
FRACTION_LABELS = ["5%", "10%", "25%", "50%", "75%", "100%"]

HEATMAP_CMAP = "YlOrRd"
FIGURE_BG    = "#FAFAF8"
PANEL_BG     = "#F2F0EC"
GRID_COLOR   = "#D8D4CC"

FONT_LABEL = {"family": "serif", "size": 10}
FONT_TICK  = {"family": "monospace", "size": 8}
FONT_ANNOT = {"family": "monospace", "size": 7}

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"RQ2 results not found: {csv_path}\n"
            "Run run_rq2_class_based.py first."
        )
    df = pd.read_csv(csv_path)

    # Support both bli_p1 (legacy) and bli_p5
    if "bli_p1" in df.columns and "bli_p5" not in df.columns:
        df = df.rename(columns={"bli_p1": "bli_p5"})

    df["language_display"] = df["language"].map(LANG_DISPLAY).fillna(df["language"])
    df["language_display"] = pd.Categorical(
        df["language_display"], categories=LANG_ORDER_DISPLAY, ordered=True
    )
    df["method"] = pd.Categorical(
        df["method"], categories=METHOD_ORDER, ordered=True
    )

    for col in ["f1", "precision", "recall", "fraction", "subset_tokens",
                "cka", "bli_p5", "en_baseline_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(
        ["language_display", "fraction", "method"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out = out_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot] saved {out}")


# ---------------------------------------------------------------------------
# Three-panel method figures
# ---------------------------------------------------------------------------

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()
    piv = df.pivot_table(
        index="language", columns="fraction",
        values=value_col, aggfunc="mean"
    )
    langs_present = [l for l in LANGUAGE_ORDER if l in piv.index]
    fracs_present = [f for f in FRACTION_ORDER if f in piv.columns]
    return piv.reindex(index=langs_present, columns=fracs_present)


def _draw_heatmap(ax, data: pd.DataFrame, title: str, norm: Normalize, cmap: str):
    if data.empty:
        ax.set_visible(False)
        return None
    ax.imshow(data.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(
        [FRACTION_LABELS[FRACTION_ORDER.index(f)] if f in FRACTION_ORDER else str(f)
         for f in data.columns],
        fontdict=FONT_TICK,
    )
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(
        [LANGUAGE_LABELS.get(l, l) for l in data.index],
        fontdict=FONT_TICK,
    )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontdict=FONT_ANNOT,
                    color="black" if val < (norm.vmax * 0.6) else "white",
                )
    ax.set_title(title, fontdict=FONT_LABEL, pad=6)
    ax.set_facecolor(PANEL_BG)


def plot_method_panels(df_method: pd.DataFrame, method: str, out_dir: Path) -> None:
    fig = plt.figure(figsize=(13, 11), facecolor=FIGURE_BG)
    fig.suptitle(
        f"RQ2 — {method} alignment\n"
        "Zero-shot NER transfer across corpus fractions",
        fontsize=14, fontweight="bold", fontfamily="serif", y=0.98,
    )

    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1], hspace=0.45)
    ax_line = fig.add_subplot(gs[0])
    ax_cka  = fig.add_subplot(gs[1])
    ax_bli  = fig.add_subplot(gs[2])

    # Panel 1 — F1 line
    ax_line.set_facecolor(PANEL_BG)
    ax_line.grid(axis="y", color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax_line.grid(axis="x", color=GRID_COLOR, linewidth=0.5, linestyle=":")

    for lang in LANGUAGE_ORDER:
        df_lang = df_method[df_method["language"] == lang].sort_values("fraction")
        if df_lang.empty:
            continue
        ax_line.plot(
            df_lang["fraction"], df_lang["f1"],
            color=LANGUAGE_COLORS[lang], linestyle="-", marker="o",
            linewidth=2, markersize=6,
            label=LANGUAGE_LABELS_FLAT[lang], zorder=3,
        )

    en_f1 = df_method["en_baseline_f1"].dropna().unique()
    if len(en_f1) > 0:
        ax_line.axhline(
            y=en_f1[0], color="#333333", linestyle=":", linewidth=1.5,
            label=f"English baseline ({en_f1[0]:.3f})", zorder=2,
        )

    ax_line.axhline(
        y=BREAKEVEN_F1, color="#888888", linestyle="--", linewidth=1,
        label=f"Break-even F1={BREAKEVEN_F1}", zorder=2,
    )

    ax_line.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)
    ax_line.set_ylabel("Entity-level F1", fontdict=FONT_LABEL)
    ax_line.set_xlim(0.02, 1.05)
    ax_line.set_ylim(bottom=0)
    ax_line.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax_line.tick_params(labelsize=8)
    ax_line.legend(
        loc="upper left", fontsize=8, framealpha=0.85,
        edgecolor=GRID_COLOR, prop={"family": "serif"},
    )
    ax_line.set_title(
        "Panel 1 — Zero-shot NER F1 vs corpus fraction",
        fontdict=FONT_LABEL, pad=8,
    )

    # Panels 2 & 3 — shared colour scale
    cka_piv = _pivot(df_method, "cka")
    bli_piv = _pivot(df_method, "bli_p5")

    all_vals = np.concatenate([
        cka_piv.values.flatten() if not cka_piv.empty else np.array([]),
        bli_piv.values.flatten() if not bli_piv.empty else np.array([]),
    ])
    all_vals = all_vals[~np.isnan(all_vals)]
    shared_vmin = float(np.nanmin(all_vals)) if len(all_vals) else 0.0
    shared_vmax = max(
        float(np.nanmax(all_vals)) * 1.05 if len(all_vals) else 0.5,
        shared_vmin + 0.01,
    )
    norm = Normalize(vmin=shared_vmin, vmax=shared_vmax)

    _draw_heatmap(
        ax_cka, cka_piv,
        "Panel 2 — CKA (alignment geometry)",
        norm, HEATMAP_CMAP,
    )
    ax_cka.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

    _draw_heatmap(
        ax_bli, bli_piv,
        "Panel 3 — BLI p@5 (lexicon induction accuracy)",
        norm, HEATMAP_CMAP,
    )
    ax_bli.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

    cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.28])
    sm = ScalarMappable(cmap=HEATMAP_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Score", fontsize=8, fontfamily="serif")

    fig.text(
        0.01, 0.22,
        "← disjunctive\n← disjunctive\n← conjunctive",
        fontsize=7, fontfamily="monospace", color="#666666", va="top",
    )

    _save(fig, out_dir, f"rq2_{method.lower()}_panels.png")


# ---------------------------------------------------------------------------
# Summary figure 1 — learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, lang in zip(axes, LANG_ORDER_DISPLAY):
        sub = df[df["language_display"] == lang]
        for method in METHOD_ORDER:
            msub = sub[sub["method"] == method].sort_values("subset_tokens")
            if msub.empty:
                continue
            x = np.log10(msub["subset_tokens"].clip(lower=1))
            y = msub["f1"]
            ax.plot(
                x, y, marker="o", label=method,
                color=METHOD_PALETTE[method], linewidth=2,
            )
            for i in range(len(y) - 1):
                if y.iloc[i] < BREAKEVEN_F1 <= y.iloc[i + 1]:
                    be_x = x.iloc[i + 1]
                    ax.axvline(be_x, color=METHOD_PALETTE[method],
                               linestyle=":", linewidth=1, alpha=0.7)
                    ax.scatter([be_x], [BREAKEVEN_F1], marker="*", s=160,
                               color=METHOD_PALETTE[method], zorder=5,
                               label="_nolegend_")

        ax.axhline(BREAKEVEN_F1, color="black", linestyle="--",
                   linewidth=1, label=f"F1={BREAKEVEN_F1}")
        ax.set_title(lang, fontweight="bold")
        ax.set_xlabel("log10(subset tokens)")
        if lang == LANG_ORDER_DISPLAY[0]:
            ax.set_ylabel("Entity-level F1")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0, top=max(df["f1"].max() * 1.2, 0.05))
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"1e{t:.0f}" for t in ticks], fontsize=8)

    fig.suptitle(
        "RQ2 — NER F1 learning curves by corpus size\n"
        f"(star = first crossing F1 >= {BREAKEVEN_F1}; "
        "dashed = break-even threshold)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir, "rq2_learning_curves.png")


# ---------------------------------------------------------------------------
# Summary figure 2 — break-even bar chart
# ---------------------------------------------------------------------------

def plot_breakeven_table(df: pd.DataFrame, out_dir: Path) -> None:
    records = []
    for lang in LANG_ORDER_DISPLAY:
        sub = df[df["language_display"] == lang]
        for method in METHOD_ORDER:
            msub = sub[sub["method"] == method].sort_values("subset_tokens")
            crossed = msub[msub["f1"] >= BREAKEVEN_F1]
            tokens = (float("nan") if crossed.empty
                      else crossed["subset_tokens"].iloc[0])
            records.append({
                "language": lang,
                "method": method,
                "breakeven_tokens": tokens,
            })

    be_df = pd.DataFrame(records)
    be_df["method"] = pd.Categorical(
        be_df["method"], categories=METHOD_ORDER, ordered=True
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.22
    x = np.arange(len(LANG_ORDER_DISPLAY))

    # Determine a sane y-range. When NO method crosses the threshold (the honest
    # case here), every bar is 0; fall back to the corpus-size range so the axis
    # and the "never" labels stay bounded (a label far outside the data range
    # blows up bbox_inches="tight").
    crossed_vals = be_df["breakeven_tokens"].dropna().tolist()
    top = max(crossed_vals) if crossed_vals else float(df["subset_tokens"].max())
    offset = top * 0.01

    for i, method in enumerate(METHOD_ORDER):
        msub = be_df[be_df["method"] == method]
        vals = [
            msub[msub["language"] == l]["breakeven_tokens"].values[0]
            for l in LANG_ORDER_DISPLAY
        ]
        numeric_vals = [v if not np.isnan(v) else 0 for v in vals]
        bars = ax.bar(
            x + (i - 1) * width, numeric_vals, width,
            label=method, color=METHOD_PALETTE[method], alpha=0.85,
        )
        for bar, raw in zip(bars, vals):
            label = "never" if np.isnan(raw) else f"{int(raw):,}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                label, ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylim(0, top * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(LANG_ORDER_DISPLAY)
    ax.set_ylabel("Corpus tokens at break-even")
    ax.set_title(
        f"RQ2 — Min tokens to reach F1 >= {BREAKEVEN_F1}\n"
        "(bars at 0 = threshold never reached with available data)",
        fontsize=13,
    )
    ax.legend(title="Method")
    fig.tight_layout()
    _save(fig, out_dir, "rq2_breakeven_table.png")


# ---------------------------------------------------------------------------
# Summary figure 3 — conjunctive vs disjunctive
# ---------------------------------------------------------------------------

def plot_conjunctive_vs_disjunctive(df: pd.DataFrame, out_dir: Path) -> None:
    df2 = df.copy()
    df2["morphology"] = df2["language"].map(
        lambda l: "Conjunctive (isiZulu)"
        if l == "zul" else "Disjunctive (Sepedi + Setswana)"
    )

    agg = (
        df2.groupby(["morphology", "fraction"], observed=True)["f1"]
        .mean()
        .reset_index()
        .sort_values("fraction")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for morpho, color in MORPHO_PALETTE.items():
        sub = agg[agg["morphology"] == morpho]
        ax.plot(
            sub["fraction"] * 100, sub["f1"],
            marker="o", label=morpho, color=color, linewidth=2,
        )

    ax.axhline(
        BREAKEVEN_F1, color="black", linestyle="--",
        linewidth=1, label=f"Break-even F1={BREAKEVEN_F1}",
    )
    ax.set_xlabel("Corpus fraction used (%)")
    ax.set_ylabel("Mean entity-level F1 (across methods)")
    ax.set_title(
        "RQ2 — Conjunctive vs Disjunctive language data efficiency\n"
        "(mean F1 across CCA / KCCA / VecMap methods)",
        fontsize=13,
    )
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, out_dir, "rq2_conjunctive_vs_disjunctive.png")


# ---------------------------------------------------------------------------
# Summary figure 4 — method heatmap
# ---------------------------------------------------------------------------

def plot_method_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    df2 = df.copy()
    df2["lang_frac"] = (
        df2["language_display"].astype(str) + "\n"
        + df2["fraction"].map(lambda f: f"{int(f * 100)}%")
    )

    col_order = []
    for lang in LANG_ORDER_DISPLAY:
        fracs = sorted(df2[df2["language_display"] == lang]["fraction"].unique())
        for f in fracs:
            col_order.append(f"{lang}\n{int(f * 100)}%")

    pivot = df2.pivot_table(
        index="method", columns="lang_frac", values="f1", observed=False
    )
    pivot = pivot.reindex(
        index=METHOD_ORDER,
        columns=[c for c in col_order if c in pivot.columns],
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.8 + 2), 4)
    )
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.0, vmax=max(df["f1"].max() * 1.2, 0.05),
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "F1"},
        ax=ax,
    )
    ax.set_title(
        "RQ2 — Zero-shot NER F1 heatmap "
        "(method x language x corpus fraction)\n"
        "green = high F1, red = low F1",
        fontsize=13,
    )
    ax.set_xlabel("Language — corpus fraction")
    ax.set_ylabel("Method")
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    fig.tight_layout()
    _save(fig, out_dir, "rq2_method_heatmap.png")


# ---------------------------------------------------------------------------
# PRIMARY RQ2 figure — intrinsic data-efficiency (decoupled from NER domain wall)
# ---------------------------------------------------------------------------

def plot_intrinsic_efficiency(df: pd.DataFrame, out_dir: Path) -> None:
    """Data-efficiency on the INTRINSIC alignment metrics (BLI p@5, CKA).

    This is the figure that actually answers RQ2: downstream NER F1 is floored
    by the CoNLL->MasakhaNER domain mismatch regardless of corpus size, so it
    cannot reveal a data-efficiency signal. BLI p@5 and post-alignment CKA
    reflect embedding/alignment quality directly and are plotted vs log-tokens.
    """
    metrics = [(c, l) for c, l in [("bli_p5", "BLI p@5 (lexicon induction)"),
                                    ("cka", "CKA (post-alignment geometry)")]
               if c in df.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(
        len(metrics), 3, figsize=(18, 5 * len(metrics)), sharex="col", squeeze=False,
    )

    for row, (col, ylabel) in enumerate(metrics):
        for ax, lang in zip(axes[row], LANG_ORDER_DISPLAY):
            sub = df[df["language_display"] == lang]
            for method in METHOD_ORDER:
                msub = sub[sub["method"] == method].sort_values("subset_tokens")
                msub = msub[msub[col].notna()]
                if msub.empty:
                    continue
                x = np.log10(msub["subset_tokens"].clip(lower=1))
                ax.plot(x, msub[col], marker="o", label=method,
                        color=METHOD_PALETTE[method], linewidth=2)
            if row == 0:
                ax.set_title(lang, fontweight="bold")
            ax.set_ylabel(ylabel if lang == LANG_ORDER_DISPLAY[0] else "")
            ax.set_xlabel("log10(subset tokens)")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"1e{t:.0f}" for t in ticks], fontsize=8)

    fig.suptitle(
        "RQ2 — Intrinsic data-efficiency: alignment quality vs corpus size\n"
        "(rising curve = more monolingual data helps; flat = data is not the bottleneck)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, "rq2_intrinsic_efficiency.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Visualise RQ2 results")
    parser.add_argument(
        "--results-csv", required=True,
        help="Path to results CSV "
             "e.g. results/rq2/full_baseline_v2/results.csv",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for plots "
             "(defaults to same folder as results CSV)",
    )
    args = parser.parse_args(argv)

    results_path = Path(args.results_csv)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_path)
    print(f"[plot] loaded {len(df)} rows from {results_path}")

    # PRIMARY RQ2 answer: intrinsic data-efficiency (decoupled from NER domain wall)
    plot_intrinsic_efficiency(df, out_dir)

    for method in df["method"].unique():
        plot_method_panels(
            df[df["method"] == method].copy(), str(method), out_dir
        )

    plot_learning_curves(df, out_dir)
    plot_breakeven_table(df, out_dir)
    plot_conjunctive_vs_disjunctive(df, out_dir)
    plot_method_heatmap(df, out_dir)

    print(f"[plot] all figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())