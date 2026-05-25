"""visualize_rq2.py

Generates one three-panel figure per alignment method:
  Panel 1 — Line plot : F1 vs corpus fraction (one line per language)
  Panel 2 — Heatmap   : CKA (languages x fractions)
  Panel 3 — Heatmap   : BLI p@5 (languages x fractions)

Languages are ordered by morphological type:
  tsn (Setswana, disjunctive)
  nso (Sepedi,   disjunctive)
  zul (isiZulu,  conjunctive)

Usage example:
    python visualize_rq2.py --results-csv results/rq2/full_baseline/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Language display order — disjunctive first, conjunctive last
LANGUAGE_ORDER = ["tsn", "nso", "zul"]

LANGUAGE_LABELS = {
    "tsn": "Setswana\n(disjunctive)",
    "nso": "Sepedi\n(disjunctive)",
    "zul": "isiZulu\n(conjunctive)",
}

LANGUAGE_COLORS = {
    "tsn": "#2A9D8F",   # teal   — disjunctive
    "nso": "#E9C46A",   # amber  — disjunctive
    "zul": "#E63946",   # red    — conjunctive
}

FRACTION_ORDER = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

FRACTION_LABELS = ["5%", "10%", "25%", "50%", "75%", "100%"]

# Shared colour scale for both heatmaps
HEATMAP_CMAP = "YlOrRd"
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 0.5   # adjust if scores exceed this after the full run

FIGURE_BG  = "#FAFAF8"
PANEL_BG   = "#F2F0EC"
GRID_COLOR = "#D8D4CC"

FONT_TITLE  = {"family": "serif", "size": 13, "weight": "bold"}
FONT_LABEL  = {"family": "serif", "size": 10}
FONT_TICK   = {"family": "monospace", "size": 8}
FONT_ANNOT  = {"family": "monospace", "size": 7}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot to (language × fraction) matrix in the display order."""
    piv = df.pivot_table(index="language", columns="fraction", values=value_col, aggfunc="mean")
    # Reindex rows and columns to display order
    langs_present = [l for l in LANGUAGE_ORDER if l in piv.index]
    fracs_present = [f for f in FRACTION_ORDER if f in piv.columns]
    return piv.reindex(index=langs_present, columns=fracs_present)


def _draw_heatmap(ax, data: pd.DataFrame, title: str, norm: Normalize, cmap: str, annotate: bool = True):
    """Draw a single heatmap panel."""
    im = ax.imshow(data.values, aspect="auto", cmap=cmap, norm=norm)

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

    # Cell annotations
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center", va="center",
                        fontdict=FONT_ANNOT,
                        color="black" if val < (HEATMAP_VMAX * 0.6) else "white",
                    )

    ax.set_title(title, fontdict=FONT_LABEL, pad=6)
    ax.set_facecolor(PANEL_BG)
    return im

# ---------------------------------------------------------------------------
# Per-method figure
# ---------------------------------------------------------------------------

def plot_method(df_method: pd.DataFrame, method: str, out_dir: Path) -> None:
    """Generate one three-panel figure for a single alignment method."""

    fig = plt.figure(figsize=(13, 11), facecolor=FIGURE_BG)
    fig.suptitle(
        f"RQ2 — {method} alignment\nZero-shot NER transfer across corpus fractions",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        y=0.98,
    )


    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1], hspace=0.45)

    ax_line  = fig.add_subplot(gs[0])
    ax_cka   = fig.add_subplot(gs[1])
    ax_bli   = fig.add_subplot(gs[2])

    # -----------------------------------------------------------------------
    # Panel 1 — F1 learning curve
    # -----------------------------------------------------------------------
    ax_line.set_facecolor(PANEL_BG)
    ax_line.grid(axis="y", color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax_line.grid(axis="x", color=GRID_COLOR, linewidth=0.5, linestyle=":")

    for lang in LANGUAGE_ORDER:
        df_lang = df_method[df_method["language"] == lang].sort_values("fraction")
        if df_lang.empty:
            continue
        color = LANGUAGE_COLORS[lang]
        label = LANGUAGE_LABELS[lang].replace("\n", " ")

        ax_line.plot(
            df_lang["fraction"],
            df_lang["f1"],
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
            zorder=3,
        )

    # English baseline
    en_f1 = df_method["en_baseline_f1"].dropna().unique()
    if len(en_f1) > 0:
        ax_line.axhline(
            y=en_f1[0],
            color="#333333",
            linestyle=":",
            linewidth=1.5,
            label=f"English baseline F1 ({en_f1[0]:.3f})",
            zorder=2,
        )

    ax_line.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)
    ax_line.set_ylabel("Entity-level F1", fontdict=FONT_LABEL)
    ax_line.set_xlim(0.02, 1.05)
    ax_line.set_ylim(bottom=0)
    ax_line.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax_line.tick_params(labelsize=8)
    ax_line.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.85,
        edgecolor=GRID_COLOR,
        prop={"family": "serif"},
    )
    ax_line.set_title("Panel 1 — Zero-shot NER F1 vs corpus fraction", fontdict=FONT_LABEL, pad=8)
    ax_line.set_facecolor(PANEL_BG)

    # -----------------------------------------------------------------------
    # Panels 2 & 3 — shared colour scale
    # -----------------------------------------------------------------------
    # Compute shared vmin/vmax across both CKA and BLI p@5
    cka_piv  = _pivot(df_method, "cka")
    bli_piv  = _pivot(df_method, "bli_p5")

    all_vals = np.concatenate([
        cka_piv.values.flatten(),
        bli_piv.values.flatten(),
    ])
    all_vals = all_vals[~np.isnan(all_vals)]
    shared_vmin = float(np.nanmin(all_vals)) if len(all_vals) else 0.0
    shared_vmax = float(np.nanmax(all_vals)) if len(all_vals) else 0.5
    # Give a small buffer above max
    shared_vmax = max(shared_vmax * 1.05, shared_vmin + 0.01)

    norm = Normalize(vmin=shared_vmin, vmax=shared_vmax)

    # Panel 2 — CKA
    im_cka = _draw_heatmap(
        ax_cka, cka_piv,
        "Panel 2 — CKA (alignment geometry)",
        norm, HEATMAP_CMAP,
    )
    ax_cka.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

    # Panel 3 — BLI p@5
    im_bli = _draw_heatmap(
        ax_bli, bli_piv,
        "Panel 3 — BLI p@5 (lexicon induction accuracy)",
        norm, HEATMAP_CMAP,
    )
    ax_bli.set_xlabel("Corpus fraction", fontdict=FONT_LABEL)

    # Shared colorbar to the right of both heatmaps
    cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.28])
    sm = ScalarMappable(cmap=HEATMAP_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Score", fontsize=8, fontfamily="serif")

    # -----------------------------------------------------------------------
    # Morphological contrast annotation
    # -----------------------------------------------------------------------
    fig.text(
        0.01, 0.22,
        "<- disjunctive\n<- disjunctive\n<- conjunctive",
        fontsize=7,
        fontfamily="monospace",
        color="#666666",
        va="top",
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = out_dir / f"rq2_{method.lower()}_panels.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_BG)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Visualise RQ2 results")
    parser.add_argument(
        "--results-csv",
        required=True,
        help="Path to results CSV e.g. results/rq2/full_baseline/results.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots (defaults to same folder as results CSV)",
    )
    args = parser.parse_args(argv)

    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"[plot] results CSV not found: {results_path}")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)

    # Coerce numeric columns — empty strings become NaN
    for col in ["f1", "precision", "recall", "cka", "bli_p5", "en_baseline_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    methods = df["method"].unique()
    for method in methods:
        df_method = df[df["method"] == method].copy()
        plot_method(df_method, method, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())