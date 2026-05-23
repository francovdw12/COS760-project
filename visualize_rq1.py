"""Visualisation of the RQ1 results (alignment strategies).

Reads `results/rq1_results.csv` produced by `run_rq1.py` and regenerates the
six figures in the `results/` folder:

  1. rq1_translation_quality.png  — P@1 / P@5 / MCS per language and method
  2. rq1_cka_before_after.png     — CKA before vs after alignment (faceted by language)
  3. rq1_cka_delta.png            — CKA change (after - before)
  4. rq1_ner_f1.png               — extrinsic zero-shot NER F1
  5. rq1_heatmap.png              — summary table of all metrics
  6. rq1_radar.png                — method-comparison radar (normalised per language)

Usage:
    python visualize_rq1.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (saves files to disk)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_ROOT

# --- Shared conventions ----------------------------------------------------

RESULTS_CSV = Path(RESULTS_ROOT) / "rq1_results.csv"

# Language codes -> display names and a fixed column/x-axis ordering.
LANG_DISPLAY = {"zul": "isiZulu", "nso": "Sepedi", "tsn": "Setswana"}
LANG_ORDER = ["isiZulu", "Sepedi", "Setswana"]

# Fixed method ordering and a palette shared across every figure.
METHOD_ORDER = ["CCA", "KCCA", "VecMap"]
METHOD_PALETTE = {
    "CCA": "#4C72B0",     # blue
    "KCCA": "#C44E52",    # red
    "VecMap": "#55A868",  # green
}

# Before/after palette for the CKA figure.
STAGE_PALETTE = {"Before": "#C44E52", "After": "#55A868"}

sns.set_theme(style="whitegrid")


# --- Loading ---------------------------------------------------------------

def load_results(csv_path: Path = RESULTS_CSV) -> pd.DataFrame:
    """Load the RQ1 CSV and prepare it for plotting."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"RQ1 results not found: {csv_path}\n"
            "Run `python run_rq1.py` first."
        )

    df = pd.read_csv(csv_path)

    # Language display names + categorical ordering.
    df["language"] = df["language"].map(LANG_DISPLAY).fillna(df["language"])
    df["language"] = pd.Categorical(df["language"], categories=LANG_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)

    # Numeric coercion (NER_F1 may be empty when the NER model was unavailable).
    for col in ["CKA_before", "CKA_after", "P@1", "P@5", "MCS", "NER_F1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["NER_F1"] = df["NER_F1"].fillna(0.0)

    # CKA change after alignment.
    df["CKA_delta"] = df["CKA_after"] - df["CKA_before"]

    return df.sort_values(["language", "method"]).reset_index(drop=True)


def _save(fig: plt.Figure, name: str) -> None:
    out = Path(RESULTS_ROOT) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --- 1. Intrinsic translation quality --------------------------------------

def plot_translation_quality(df: pd.DataFrame) -> None:
    metrics = ["P@1", "P@5", "MCS"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=df, x="language", y=metric, hue="method",
            hue_order=METHOD_ORDER, palette=METHOD_PALETTE, ax=ax,
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.legend(title="Method")

    fig.suptitle(
        "RQ1 — Intrinsic word-translation quality\n"
        "P@1 / P@5 : nearest-neighbour accuracy  |  MCS : mean cosine similarity",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "rq1_translation_quality.png")


# --- 2. CKA before / after -------------------------------------------------

def plot_cka_before_after(df: pd.DataFrame) -> None:
    long = df.melt(
        id_vars=["language", "method"],
        value_vars=["CKA_before", "CKA_after"],
        var_name="Stage", value_name="CKA",
    )
    long["Stage"] = long["Stage"].map({"CKA_before": "Before", "CKA_after": "After"})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, lang in zip(axes, LANG_ORDER):
        sns.barplot(
            data=long[long["language"] == lang],
            x="method", y="CKA", hue="Stage",
            order=METHOD_ORDER, hue_order=["Before", "After"],
            palette=STAGE_PALETTE, ax=ax,
        )
        ax.set_title(lang, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("CKA" if lang == LANG_ORDER[0] else "")
        ax.legend(title="Stage")

    fig.suptitle(
        "RQ1 — CKA geometric similarity before vs after alignment\n"
        "Lower CKA_after means the method distorts the global space",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "rq1_cka_before_after.png")


# --- 3. CKA change ---------------------------------------------------------

def plot_cka_delta(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df, x="language", y="CKA_delta", hue="method",
        hue_order=METHOD_ORDER, palette=METHOD_PALETTE, ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("ΔCKA")
    ax.legend(title="Method")
    ax.set_title(
        "RQ1 — CKA change after alignment  (CKA_after − CKA_before)\n"
        "Positive = geometry improved  |  Negative = space distorted",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "rq1_cka_delta.png")


# --- 4. Extrinsic NER F1 ---------------------------------------------------

def plot_ner_f1(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df, x="language", y="NER_F1", hue="method",
        hue_order=METHOD_ORDER, palette=METHOD_PALETTE, ax=ax,
    )
    top = max(0.10, float(df["NER_F1"].max()) * 1.15)
    ax.set_ylim(0.0, top)
    ax.set_xlabel("")
    ax.set_ylabel("Entity-level F1")
    ax.legend(title="Method")
    ax.set_title(
        "RQ1 — Extrinsic: zero-shot NER F1\n"
        "(BiLSTM-CRF trained on CoNLL-2003 → evaluated on MasakhaNER)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "rq1_ner_f1.png")


# --- 5. Summary heatmap ----------------------------------------------------

def plot_heatmap(df: pd.DataFrame) -> None:
    metrics = ["P@1", "P@5", "MCS", "CKA_before", "CKA_after", "CKA_delta", "NER_F1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        pivot = df.pivot_table(index="method", values=metric, columns="language", observed=False)
        pivot = pivot.reindex(index=METHOD_ORDER, columns=LANG_ORDER)
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            cbar=False, linewidths=0.5, linecolor="white", ax=ax,
        )
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        if i != 0:
            ax.set_yticklabels([])
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("RQ1 — Complete results summary (green = good, red = poor)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "rq1_heatmap.png")


# --- 6. Comparison radar ---------------------------------------------------

def plot_radar(df: pd.DataFrame) -> None:
    categories = ["P@1", "P@5", "MCS"]
    n = len(categories)
    angles = [k / n * 2 * np.pi for k in range(n)] + [0.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"polar": True})

    for ax, lang in zip(axes, LANG_ORDER):
        sub = df[df["language"] == lang]

        # Min-max normalisation per language and per metric (MCS can be negative).
        norm = {}
        for metric in categories:
            vals = sub.set_index("method")[metric].reindex(METHOD_ORDER).astype(float)
            lo, hi = vals.min(), vals.max()
            span = hi - lo
            norm[metric] = (vals - lo) / span if span > 1e-12 else vals * 0.0 + 0.5

        for method in METHOD_ORDER:
            values = [norm[m][method] for m in categories]
            values += values[:1]
            ax.plot(angles, values, label=method, color=METHOD_PALETTE[method], linewidth=1.5)
            ax.fill(angles, values, color=METHOD_PALETTE[method], alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        ax.set_title(lang, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)

    fig.suptitle(
        "RQ1 — Method comparison radar (normalised per language)\n"
        "Larger area = better overall alignment",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "rq1_radar.png")


# --- Entry point -----------------------------------------------------------

def main() -> None:
    df = load_results()
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")
    plot_translation_quality(df)
    plot_cka_before_after(df)
    plot_cka_delta(df)
    plot_ner_f1(df)
    plot_heatmap(df)
    plot_radar(df)
    print("All RQ1 figures regenerated.")


if __name__ == "__main__":
    main()
