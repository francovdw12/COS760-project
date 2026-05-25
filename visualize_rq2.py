"""Visualisation of the RQ2 results (data efficiency / learning curves).

Reads `results/rq2_results.csv` produced by `run_rq2.py` and regenerates
four figures in the `results/` folder:

  1. rq2_learning_curves.png      — NER F1 vs corpus size per language/method
  2. rq2_breakeven_table.png      — min tokens to reach F1 ≥ 0.50 per method
  3. rq2_conjunctive_vs_disjunctive.png — isiZulu vs Sepedi+Setswana mean F1
  4. rq2_method_heatmap.png       — F1 heatmap (method × language×fraction)

Usage:
    python visualize_rq2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_ROOT

RESULTS_CSV = Path(RESULTS_ROOT) / "rq2_results.csv"
BREAKEVEN_F1 = 0.50

LANG_DISPLAY = {"zul": "isiZulu", "nso": "Sepedi", "tsn": "Setswana"}
LANG_ORDER = ["isiZulu", "Sepedi", "Setswana"]

METHOD_ORDER = ["CCA", "KCCA", "VecMap"]
METHOD_PALETTE = {
    "CCA": "#4C72B0",
    "KCCA": "#C44E52",
    "VecMap": "#55A868",
}

MORPHO_PALETTE = {
    "Conjunctive (isiZulu)": "#E07B54",
    "Disjunctive (Sepedi + Setswana)": "#5BA4CF",
}

sns.set_theme(style="whitegrid")


def load_results(csv_path: Path = RESULTS_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"RQ2 results not found: {csv_path}\n"
            "Run `python run_rq2.py` first."
        )
    df = pd.read_csv(csv_path)
    df["language_display"] = df["language"].map(LANG_DISPLAY).fillna(df["language"])
    df["language_display"] = pd.Categorical(
        df["language_display"], categories=LANG_ORDER, ordered=True
    )
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    for col in ["f1", "precision", "recall", "fraction", "subset_tokens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["language_display", "fraction", "method"]).reset_index(drop=True)


def _save(fig: plt.Figure, name: str) -> None:
    out = Path(RESULTS_ROOT) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --- 1. Learning curves -------------------------------------------------------

def plot_learning_curves(df: pd.DataFrame) -> None:
    """One panel per language; x = log10(subset_tokens), y = F1; one line/method."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, lang in zip(axes, LANG_ORDER):
        sub = df[df["language_display"] == lang]
        for method in METHOD_ORDER:
            msub = sub[sub["method"] == method].sort_values("subset_tokens")
            if msub.empty:
                continue
            x = np.log10(msub["subset_tokens"].clip(lower=1))
            y = msub["f1"]
            ax.plot(x, y, marker="o", label=method,
                    color=METHOD_PALETTE[method], linewidth=2)

            # Mark break-even crossing
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
        if lang == LANG_ORDER[0]:
            ax.set_ylabel("Entity-level F1")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.02)

        # Secondary x-axis labels: actual token counts
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"1e{t:.0f}" for t in ticks], fontsize=8)

    fig.suptitle(
        "RQ2 — NER F1 learning curves by corpus size\n"
        f"(star marker = first crossing F1 >= {BREAKEVEN_F1}; dashed = break-even threshold)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "rq2_learning_curves.png")


# --- 2. Break-even bar chart --------------------------------------------------

def plot_breakeven_table(df: pd.DataFrame) -> None:
    """Bar chart: min tokens to reach F1 ≥ BREAKEVEN_F1, per method per language."""
    records = []
    for lang in LANG_ORDER:
        sub = df[df["language_display"] == lang]
        for method in METHOD_ORDER:
            msub = sub[sub["method"] == method].sort_values("subset_tokens")
            crossed = msub[msub["f1"] >= BREAKEVEN_F1]
            if crossed.empty:
                tokens = float("nan")
            else:
                tokens = crossed["subset_tokens"].iloc[0]
            records.append({"language": lang, "method": method, "breakeven_tokens": tokens})

    be_df = pd.DataFrame(records)
    be_df["method"] = pd.Categorical(be_df["method"], categories=METHOD_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    n_methods = len(METHOD_ORDER)
    n_langs = len(LANG_ORDER)
    width = 0.22
    x = np.arange(n_langs)

    for i, method in enumerate(METHOD_ORDER):
        msub = be_df[be_df["method"] == method]
        vals = [msub[msub["language"] == l]["breakeven_tokens"].values[0] for l in LANG_ORDER]
        numeric_vals = [v if not np.isnan(v) else 0 for v in vals]
        bars = ax.bar(x + (i - 1) * width, numeric_vals, width,
                      label=method, color=METHOD_PALETTE[method], alpha=0.85)
        for j, (bar, raw) in enumerate(zip(bars, vals)):
            label = "never" if np.isnan(raw) else f"{int(raw):,}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 500,
                    label, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(LANG_ORDER)
    ax.set_ylabel("Corpus tokens at break-even")
    ax.set_title(
        f"RQ2 — Min tokens to reach F1 ≥ {BREAKEVEN_F1}\n"
        "(bars at 0 = never reached with available data)",
        fontsize=13,
    )
    ax.legend(title="Method")
    fig.tight_layout()
    _save(fig, "rq2_breakeven_table.png")


# --- 3. Conjunctive vs disjunctive -------------------------------------------

def plot_conjunctive_vs_disjunctive(df: pd.DataFrame) -> None:
    """Mean F1 per fraction: isiZulu (conjunctive) vs Sepedi+Setswana (disjunctive)."""
    df2 = df.copy()
    df2["morphology"] = df2["language"].map(
        lambda l: "Conjunctive (isiZulu)" if l == "zul" else "Disjunctive (Sepedi + Setswana)"
    )

    # Mean F1 across methods and languages within each morphology group per fraction
    agg = (
        df2.groupby(["morphology", "fraction"], observed=True)["f1"]
        .mean()
        .reset_index()
        .sort_values("fraction")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for morpho, color in MORPHO_PALETTE.items():
        sub = agg[agg["morphology"] == morpho]
        x = np.log10(sub["fraction"].clip(lower=1e-6) * 1e6)  # approx scale
        ax.plot(
            sub["fraction"] * 100, sub["f1"],
            marker="o", label=morpho, color=color, linewidth=2,
        )

    ax.axhline(BREAKEVEN_F1, color="black", linestyle="--",
               linewidth=1, label=f"Break-even F1={BREAKEVEN_F1}")
    ax.set_xlabel("Corpus fraction used (%)")
    ax.set_ylabel("Mean entity-level F1 (across methods)")
    ax.set_title(
        "RQ2 — Conjunctive vs Disjunctive language data efficiency\n"
        "(mean F1 across CCA / KCCA / VecMap methods)",
        fontsize=13,
    )
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, "rq2_conjunctive_vs_disjunctive.png")


# --- 4. Method heatmap -------------------------------------------------------

def plot_method_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of F1: rows = method, columns = (language, fraction)."""
    df2 = df.copy()
    df2["lang_frac"] = df2["language_display"].astype(str) + "\n" + df2["fraction"].map(
        lambda f: f"{int(f * 100)}%"
    )

    # Order columns by language then fraction
    col_order = []
    for lang in LANG_ORDER:
        fracs = sorted(df2[df2["language_display"] == lang]["fraction"].unique())
        for f in fracs:
            col_order.append(f"{lang}\n{int(f * 100)}%")

    pivot = df2.pivot_table(index="method", columns="lang_frac", values="f1", observed=False)
    pivot = pivot.reindex(index=METHOD_ORDER, columns=[c for c in col_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.8 + 2), 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "F1"},
        ax=ax,
    )
    ax.set_title(
        "RQ2 — Zero-shot NER F1 heatmap (method × language × corpus fraction)\n"
        "green = high F1, red = low F1",
        fontsize=13,
    )
    ax.set_xlabel("Language — corpus fraction")
    ax.set_ylabel("Method")
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    fig.tight_layout()
    _save(fig, "rq2_method_heatmap.png")


# --- Entry point -------------------------------------------------------------

def main() -> None:
    df = load_results()
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")
    plot_learning_curves(df)
    plot_breakeven_table(df)
    plot_conjunctive_vs_disjunctive(df)
    plot_method_heatmap(df)
    print("All RQ2 figures regenerated.")


if __name__ == "__main__":
    main()
