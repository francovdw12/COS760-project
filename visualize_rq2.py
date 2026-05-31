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
    rq2_bli_curves.png
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

# Arbitrarily set to 0.05 for visual comparison purposes only.
# Does not represent a principled performance criterion from the literature.
BREAKEVEN_F1 = 0.05

# Visual comparison ceiling for F1 plots and heatmaps.
# Set to match BREAKEVEN_F1 so all results visually read as underperforming.
F1_VMAX = 0.05

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

# ---------------------------------------------------------------------------
# Font sizes — adjust these to match the font size used in the report
# ---------------------------------------------------------------------------
FONT_SIZES = {
    "title_fig":    22,   # figure suptitle
    "title_panel":  18,   # individual panel/axes titles
    "axis_label":   16,   # x/y axis labels
    "tick":         14,   # axis tick labels
    "legend":       11,   # legend text
    "annot":        13,   # heatmap cell annotations
    "cbar_label":   14,   # colourbar labels
    "cbar_tick":    12,   # colourbar tick labels
    "morpho_note":  12,   # small morphology annotation text
    "heatmap_x":    11,   # method heatmap x-axis ticks
}

FONT_FAMILY_SERIF   = "serif"
FONT_FAMILY_MONO    = "monospace"

# Derived font dicts used by matplotlib fontdict= arguments
FONT_LABEL = {"family": FONT_FAMILY_SERIF, "size": FONT_SIZES["axis_label"]}
FONT_TICK  = {"family": FONT_FAMILY_MONO,  "size": FONT_SIZES["tick"]}
FONT_ANNOT = {"family": FONT_FAMILY_MONO,  "size": FONT_SIZES["annot"]}

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
                "cka", "bli_p5", "en_baseline_f1", "cka_pre", "n_train_anchors"]:
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


def _draw_heatmap(ax, data: pd.DataFrame, title: str, norm: Normalize, cmap: str,
                  annot_df: pd.DataFrame | None = None):
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
                cell_text = f"{val:.3f}"
                if annot_df is not None and not annot_df.empty:
                    try:
                        anc = annot_df.values[i, j]
                        if not np.isnan(anc):
                            cell_text += f"\n(n={int(anc)})"
                    except (IndexError, ValueError):
                        pass
                ax.text(
                    j, i, cell_text,
                    ha="center", va="center",
                    fontdict=FONT_ANNOT,
                    color="black" if val < (norm.vmax * 0.6) else "white",
                )
    ax.set_title(title, fontdict={"family": FONT_FAMILY_SERIF, "size": FONT_SIZES["title_panel"]}, pad=6)
    ax.set_facecolor(PANEL_BG)


def plot_method_panels(df_method: pd.DataFrame, method: str, out_dir: Path) -> None:
    is_vecmap = method == "VecMap"

    if is_vecmap:
        fig = plt.figure(figsize=(20, 14), facecolor=FIGURE_BG)
        fig.suptitle(
            f"RQ2 — {method} alignment\n"
            "Zero-shot NER transfer across corpus fractions",
            fontsize=FONT_SIZES["title_fig"], fontweight="bold", fontfamily=FONT_FAMILY_SERIF, y=0.98,
        )
        gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1], hspace=0.45)
        ax_line     = fig.add_subplot(gs[0])
        ax_coverage = fig.add_subplot(gs[1])
        ax_bli      = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(20, 20), facecolor=FIGURE_BG)
        fig.suptitle(
            f"RQ2 — {method} alignment\n"
            "Zero-shot NER transfer across corpus fractions",
            fontsize=FONT_SIZES["title_fig"], fontweight="bold", fontfamily=FONT_FAMILY_SERIF, y=0.98,
        )
        gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 1, 1], hspace=0.45)
        ax_line     = fig.add_subplot(gs[0])
        ax_cka_pre  = fig.add_subplot(gs[1])
        ax_cka_post = fig.add_subplot(gs[2])
        ax_bli      = fig.add_subplot(gs[3])

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
        label=f"Reference F1={BREAKEVEN_F1} (arbitrary)", zorder=2,
    )

    ax_line.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)
    ax_line.set_ylabel("Entity-level F1", fontdict=FONT_LABEL)
    ax_line.set_xlim(0.02, 1.05)
    ax_line.set_ylim(0, F1_VMAX)
    ax_line.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax_line.tick_params(labelsize=FONT_SIZES["tick"])
    ax_line.legend(
        loc="upper left", fontsize=FONT_SIZES["legend"], framealpha=0.85,
        edgecolor=GRID_COLOR, prop={"family": FONT_FAMILY_SERIF},
    )
    ax_line.set_title(
        "Panel 1 — Zero-shot NER F1 vs corpus fraction",
        fontdict=FONT_LABEL, pad=8,
    )

    if is_vecmap:
        cov_piv = _pivot(df_method, "vecmap_coverage")
        bli_piv = _pivot(df_method, "bli_p5")

        all_vals = np.concatenate([
            cov_piv.values.flatten() if not cov_piv.empty else np.array([]),
            bli_piv.values.flatten() if not bli_piv.empty else np.array([]),
        ])
        all_vals = all_vals[~np.isnan(all_vals)]
        shared_vmin = float(np.nanmin(all_vals)) if len(all_vals) else 0.0
        shared_vmax = max(float(np.nanmax(all_vals)) * 1.05 if len(all_vals) else 1.0,
                          shared_vmin + 0.01)
        norm = Normalize(vmin=shared_vmin, vmax=shared_vmax)

        _draw_heatmap(ax_coverage, cov_piv,
                      "Panel 2 — Vocabulary coverage (fraction of NER tokens found in aligned vocab)",
                      norm, HEATMAP_CMAP)
        ax_coverage.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

        _draw_heatmap(ax_bli, bli_piv,
                      "Panel 3 — BLI p@5 (lexicon induction accuracy)",
                      norm, HEATMAP_CMAP)
        ax_bli.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

        cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.28])
        sm = ScalarMappable(cmap=HEATMAP_CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=FONT_SIZES["cbar_tick"])
        cbar.set_label("Score", fontsize=FONT_SIZES["cbar_label"], fontfamily=FONT_FAMILY_SERIF)

    else:
        cka_pre_piv  = _pivot(df_method, "cka_pre")
        cka_post_piv = _pivot(df_method, "cka")
        bli_piv      = _pivot(df_method, "bli_p5")
        anc_piv      = _pivot(df_method, "n_train_anchors")

        all_cka = np.concatenate([
            cka_pre_piv.values.flatten()  if not cka_pre_piv.empty  else np.array([]),
            cka_post_piv.values.flatten() if not cka_post_piv.empty else np.array([]),
        ])
        all_cka = all_cka[~np.isnan(all_cka)]
        cka_vmin = float(np.nanmin(all_cka)) if len(all_cka) else 0.0
        cka_vmax = max(float(np.nanmax(all_cka)) * 1.05 if len(all_cka) else 0.5,
                       cka_vmin + 0.01)
        cka_norm = Normalize(vmin=cka_vmin, vmax=cka_vmax)

        bli_vals = bli_piv.values.flatten() if not bli_piv.empty else np.array([])
        bli_vals = bli_vals[~np.isnan(bli_vals)]
        bli_vmin = float(np.nanmin(bli_vals)) if len(bli_vals) else 0.0
        bli_vmax = max(float(np.nanmax(bli_vals)) * 1.05 if len(bli_vals) else 0.1,
                       bli_vmin + 0.01)
        bli_norm = Normalize(vmin=bli_vmin, vmax=bli_vmax)

        _draw_heatmap(ax_cka_pre, cka_pre_piv,
                      "Panel 2 — CKA before alignment (raw space similarity)",
                      cka_norm, HEATMAP_CMAP)
        ax_cka_pre.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

        _draw_heatmap(ax_cka_post, cka_post_piv,
                      "Panel 3 — CKA after alignment (n_train_anchors in parentheses)",
                      cka_norm, HEATMAP_CMAP, annot_df=anc_piv)
        ax_cka_post.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

        _draw_heatmap(ax_bli, bli_piv,
                      "Panel 4 — BLI p@5 (lexicon induction accuracy)",
                      bli_norm, HEATMAP_CMAP)
        ax_bli.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

        cbar_cka_ax = fig.add_axes([0.92, 0.35, 0.02, 0.28])
        sm_cka = ScalarMappable(cmap=HEATMAP_CMAP, norm=cka_norm)
        sm_cka.set_array([])
        cbar_cka = fig.colorbar(sm_cka, cax=cbar_cka_ax)
        cbar_cka.ax.tick_params(labelsize=FONT_SIZES["cbar_tick"])
        cbar_cka.set_label("CKA", fontsize=FONT_SIZES["cbar_label"], fontfamily=FONT_FAMILY_SERIF)

        cbar_bli_ax = fig.add_axes([0.92, 0.05, 0.02, 0.18])
        sm_bli = ScalarMappable(cmap=HEATMAP_CMAP, norm=bli_norm)
        sm_bli.set_array([])
        cbar_bli = fig.colorbar(sm_bli, cax=cbar_bli_ax)
        cbar_bli.ax.tick_params(labelsize=FONT_SIZES["cbar_tick"])
        cbar_bli.set_label("BLI p@5", fontsize=FONT_SIZES["cbar_label"], fontfamily=FONT_FAMILY_SERIF)

    fig.text(
        0.01, 0.22,
        "← disjunctive\n← disjunctive\n← conjunctive",
        fontsize=FONT_SIZES["morpho_note"], fontfamily=FONT_FAMILY_MONO, color="#666666", va="top",
    )

    _save(fig, out_dir, f"rq2_{method.lower()}_panels.png")


# ---------------------------------------------------------------------------
# Summary figure 1 — learning curves
# ---------------------------------------------------------------------------

def _token_fmt(x, _pos=None) -> str:
    """Format token counts as e.g. 75K, 1.5M."""
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}K"
    return str(int(x))


def plot_learning_curves(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

    for ax, lang in zip(axes, LANG_ORDER_DISPLAY):
        sub = df[df["language_display"] == lang]
        for method in METHOD_ORDER:
            msub = sub[sub["method"] == method].sort_values("subset_tokens")
            if msub.empty:
                continue
            x = msub["subset_tokens"].clip(lower=1)
            y = msub["f1"]
            ax.plot(
                np.log10(x), y, marker="o", label=method,
                color=METHOD_PALETTE[method], linewidth=2,
            )
            for i in range(len(y) - 1):
                if y.iloc[i] < BREAKEVEN_F1 <= y.iloc[i + 1]:
                    be_x = np.log10(x.iloc[i + 1])
                    ax.axvline(be_x, color=METHOD_PALETTE[method],
                               linestyle=":", linewidth=1, alpha=0.7)
                    ax.scatter([be_x], [BREAKEVEN_F1], marker="*", s=160,
                               color=METHOD_PALETTE[method], zorder=5,
                               label="_nolegend_")

        ax.axhline(BREAKEVEN_F1, color="black", linestyle="--",
                   linewidth=1, label=f"Reference F1={BREAKEVEN_F1} (arbitrary)")
        ax.set_title(lang, fontweight="bold", fontsize=FONT_SIZES["title_panel"])
        ax.set_xlabel("Corpus size (tokens)", fontsize=FONT_SIZES["axis_label"])
        if lang == LANG_ORDER_DISPLAY[0]:
            ax.set_ylabel("Entity-level F1", fontsize=FONT_SIZES["axis_label"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.set_ylim(0, 0.06)

        log_ticks = ax.get_xticks()
        token_ticks = [t for t in log_ticks if 4 <= t <= 8]
        ax.set_xticks(token_ticks)
        ax.set_xticklabels([_token_fmt(10**t) for t in token_ticks], fontsize=FONT_SIZES["tick"])
        ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])

    fig.suptitle(
        "RQ2 — NER F1 learning curves by corpus size\n"
        f"(dashed = arbitrary reference F1={BREAKEVEN_F1})",
        fontsize=FONT_SIZES["title_fig"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir, "rq2_learning_curves.png")


# ---------------------------------------------------------------------------
# Summary figure 1b — BLI p@5 learning curves
# ---------------------------------------------------------------------------

def plot_bli_curves(df: pd.DataFrame, out_dir: Path) -> None:
    if "bli_p5" not in df.columns:
        print("[plot] bli_p5 column not found — skipping BLI curves")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

    # y ceiling: data max with 20% headroom, minimum 0.02 so axis is readable
    bli_max = df["bli_p5"].dropna().max()
    y_ceil = max(bli_max * 1.2, 0.02)

    for ax, lang in zip(axes, LANG_ORDER_DISPLAY):
        sub = df[df["language_display"] == lang]
        for method in METHOD_ORDER:
            msub = sub[sub["method"] == method].sort_values("subset_tokens")
            if msub.empty:
                continue
            x = msub["subset_tokens"].clip(lower=1)
            y = msub["bli_p5"].fillna(0)
            ax.plot(
                np.log10(x), y, marker="o", label=method,
                color=METHOD_PALETTE[method], linewidth=2,
            )

        ax.set_title(lang, fontweight="bold", fontsize=FONT_SIZES["title_panel"])
        ax.set_xlabel("Corpus size (tokens)", fontsize=FONT_SIZES["axis_label"])
        if lang == LANG_ORDER_DISPLAY[0]:
            ax.set_ylabel("BLI precision@5", fontsize=FONT_SIZES["axis_label"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.set_ylim(0, y_ceil)

        log_ticks = ax.get_xticks()
        token_ticks = [t for t in log_ticks if 4 <= t <= 8]
        ax.set_xticks(token_ticks)
        ax.set_xticklabels([_token_fmt(10**t) for t in token_ticks], fontsize=FONT_SIZES["tick"])
        ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])

    fig.suptitle(
        "RQ2 — BLI p@5 by corpus size\n"
        "(near-zero across all methods confirms alignment failure is not data-quantity dependent)",
        fontsize=FONT_SIZES["title_fig"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir, "rq2_bli_curves.png")


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

    fig, ax = plt.subplots(figsize=(26, 10))
    width = 0.22
    x = np.arange(len(LANG_ORDER_DISPLAY))

    max_val = be_df["breakeven_tokens"].replace(float("nan"), 0).max()
    if max_val == 0:
        max_val = 1  # fallback when no method ever crossed the threshold

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
            if np.isnan(raw):
                ax.bar(
                    bar.get_x(), max_val * 0.08 if max_val > 0 else 1,
                    width, bottom=0,
                    color="none", edgecolor="#CC3333",
                    linewidth=1.2, hatch="///",
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max_val * 0.09 if max_val > 0 else 1.1,
                    "never", ha="center", va="bottom",
                    fontsize=FONT_SIZES["tick"], color="#CC3333",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.01,
                    f"{int(raw):,}", ha="center", va="bottom", fontsize=FONT_SIZES["tick"],
                )

    ax.set_ylim(0, max_val * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(LANG_ORDER_DISPLAY)
    ax.set_ylabel("Corpus tokens at reference F1")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: _token_fmt(v)))
    ax.set_title(
        f"RQ2 — Min tokens to reach arbitrary reference F1 >= {BREAKEVEN_F1}\n"
        "(hatched red = threshold never reached within available data)",
        fontsize=FONT_SIZES["title_fig"],
    )
    ax.legend(title="Method")
    fig.tight_layout()
    _save(fig, out_dir, "rq2_breakeven_table.png")


# ---------------------------------------------------------------------------
# Summary figure 3 — conjunctive vs disjunctive (faceted by method)
# ---------------------------------------------------------------------------

def plot_conjunctive_vs_disjunctive(df: pd.DataFrame, out_dir: Path) -> None:
    df2 = df.copy()
    df2["morphology"] = df2["language"].map(
        lambda l: "Conjunctive (isiZulu)"
        if l == "zul" else "Disjunctive (Sepedi + Setswana)"
    )

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

    for ax, method in zip(axes, METHOD_ORDER):
        msub = df2[df2["method"] == method]
        agg = (
            msub.groupby(["morphology", "fraction"], observed=True)["f1"]
            .mean()
            .reset_index()
            .sort_values("fraction")
        )
        for morpho, color in MORPHO_PALETTE.items():
            sub = agg[agg["morphology"] == morpho]
            ax.plot(
                sub["fraction"] * 100, sub["f1"],
                marker="o", label=morpho, color=color, linewidth=2,
            )
        ax.axhline(
            BREAKEVEN_F1, color="black", linestyle="--",
            linewidth=1, label=f"Reference F1={BREAKEVEN_F1} (arbitrary)",
        )
        ax.set_title(method, fontweight="bold", fontsize=FONT_SIZES["title_panel"])
        ax.set_xlabel("Corpus fraction used (%)", fontsize=FONT_SIZES["axis_label"])
        if method == METHOD_ORDER[0]:
            ax.set_ylabel("Mean entity-level F1", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylim(0, F1_VMAX)
        if method == METHOD_ORDER[-1]:
            ax.legend(fontsize=FONT_SIZES["legend"], loc="upper right")

    fig.suptitle(
        "RQ2 — Conjunctive vs Disjunctive language data efficiency\n"
        "(mean F1 per method; faceted by alignment method)",
        fontsize=FONT_SIZES["title_fig"],
    )
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

    n_fracs = len(FRACTION_ORDER)
    vlines = [n_fracs * i for i in range(1, len(LANG_ORDER_DISPLAY))]

    fig, ax = plt.subplots(
        figsize=(max(16, len(pivot.columns) * 1.2 + 2), 6)
    )
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.0, vmax=F1_VMAX,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "F1"},
        ax=ax,
    )
    for vl in vlines:
        ax.axvline(vl, color="black", linewidth=1.5)

    ax.set_title(
        "RQ2 — Zero-shot NER F1 heatmap "
        "(method x language x corpus fraction)\n"
        "green = high F1, red = low F1; colour scale ceiling = arbitrary reference F1",
        fontsize=FONT_SIZES["title_fig"],
    )
    ax.set_xlabel("Language — corpus fraction", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Method", fontsize=FONT_SIZES["axis_label"])
    ax.tick_params(axis="x", labelsize=FONT_SIZES["heatmap_x"], rotation=45)
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
    plot_bli_curves(df, out_dir)
    plot_breakeven_table(df, out_dir)
    plot_conjunctive_vs_disjunctive(df, out_dir)
    plot_method_heatmap(df, out_dir)

    print(f"[plot] all figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())