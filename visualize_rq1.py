import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

LANG_NAMES  = {"zul": "isiZulu", "nso": "Sepedi", "tsn": "Setswana"}
METHOD_PALETTE = {"CCA": "#4878CF", "KCCA": "#D65F5F", "VecMap": "#6ACC65"}
LANG_ORDER  = ["isiZulu", "Sepedi", "Setswana"]
METHOD_ORDER = ["CCA", "KCCA", "VecMap"]


def _save(path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")


def plot_results(csv_path="results/rq1_results.csv"):
    df = pd.read_csv(csv_path)
    df["NER_F1"]  = pd.to_numeric(df["NER_F1"], errors="coerce")
    df["lang"]    = df["language"].map(LANG_NAMES)
    df["CKA_delta"] = df["CKA_after"] - df["CKA_before"]

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ------------------------------------------------------------------
    # Figure 1 — Word-translation quality  (P@1, P@5, MCS)
    # ylim is dynamic so negative MCS (KCCA) shows correctly
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["P@1", "P@5", "MCS"]):
        sns.barplot(
            data=df, x="lang", y=metric, hue="method",
            hue_order=METHOD_ORDER, order=LANG_ORDER,
            palette=METHOD_PALETTE, ax=ax,
        )
        ymin = min(0, df[metric].min() - 0.05)
        ymax = max(df[metric].max() + 0.05, 0.1)
        ax.set_ylim(ymin, ymax)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.legend(title="Method", fontsize=9)

    fig.suptitle(
        "RQ1 — Intrinsic word-translation quality\n"
        "P@1 / P@5 : nearest-neighbour accuracy  |  MCS : mean cosine similarity",
        fontsize=13,
    )
    plt.tight_layout()
    _save("results/rq1_translation_quality.png")
    plt.show()

    # ------------------------------------------------------------------
    # Figure 2 — CKA before vs after, one subplot per language
    # ------------------------------------------------------------------
    cka_rows = []
    for _, row in df.iterrows():
        cka_rows += [
            {"Language": row["lang"], "Method": row["method"], "Stage": "Before", "CKA": row["CKA_before"]},
            {"Language": row["lang"], "Method": row["method"], "Stage": "After",  "CKA": row["CKA_after"]},
        ]
    cka_df = pd.DataFrame(cka_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, lang in zip(axes, LANG_ORDER):
        sub = cka_df[cka_df["Language"] == lang]
        sns.barplot(
            data=sub, x="Method", y="CKA", hue="Stage",
            order=METHOD_ORDER,
            palette={"Before": "#d62728", "After": "#2ca02c"},
            ax=ax,
        )
        ax.set_title(lang, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.legend(title="Stage", fontsize=9)

    fig.suptitle(
        "RQ1 — CKA geometric similarity before vs after alignment\n"
        "Lower CKA_after means the method distorts the global space",
        fontsize=13,
    )
    plt.tight_layout()
    _save("results/rq1_cka_before_after.png")
    plt.show()

    # ------------------------------------------------------------------
    # Figure 3 — CKA delta  (CKA_after − CKA_before)
    # Positive = alignment improved geometry, negative = distortion
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df, x="lang", y="CKA_delta", hue="method",
        hue_order=METHOD_ORDER, order=LANG_ORDER,
        palette=METHOD_PALETTE, ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_title(
        "RQ1 — CKA change after alignment  (CKA_after − CKA_before)\n"
        "Positive = geometry improved  |  Negative = space distorted",
        fontsize=13,
    )
    ax.set_xlabel("")
    ax.set_ylabel("ΔCKA")
    ax.legend(title="Method")
    plt.tight_layout()
    _save("results/rq1_cka_delta.png")
    plt.show()

    # ------------------------------------------------------------------
    # Figure 4 — Full heatmap  (method × language for every metric)
    # ------------------------------------------------------------------
    metrics = ["P@1", "P@5", "MCS", "CKA_before", "CKA_after", "CKA_delta", "NER_F1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 4))

    for ax, metric in zip(axes, metrics):
        pivot = df.pivot(index="method", columns="lang", values=metric)
        pivot = pivot.reindex(index=METHOD_ORDER, columns=LANG_ORDER)
        vmin = pivot.values[np.isfinite(pivot.values)].min() if pivot.notna().any().any() else 0
        vmax = pivot.values[np.isfinite(pivot.values)].max() if pivot.notna().any().any() else 1
        center = 0 if vmin < 0 else None
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap="RdYlGn", center=center,
            vmin=vmin, vmax=vmax,
            linewidths=0.5, cbar=False,
            annot_kws={"size": 9},
        )
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("RQ1 — Complete results summary (green = good, red = poor)", fontsize=13)
    plt.tight_layout()
    _save("results/rq1_heatmap.png")
    plt.show()

    # ------------------------------------------------------------------
    # Figure 5 — Method comparison radar  (one per language)
    # Normalise each metric to [0,1] so all axes are comparable
    # ------------------------------------------------------------------
    radar_metrics = ["P@1", "P@5", "MCS"]
    # Only include metrics where there is variation
    radar_metrics = [m for m in radar_metrics if df[m].std() > 1e-6]
    if len(radar_metrics) >= 3:
        N = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5),
            subplot_kw={"projection": "polar"},
        )
        for ax, lang in zip(axes, LANG_ORDER):
            sub = df[df["lang"] == lang].set_index("method")
            # Normalise to [0,1] across all methods for this language
            norms = {}
            for m in radar_metrics:
                col = sub[m]
                lo, hi = col.min(), col.max()
                norms[m] = (col - lo) / (hi - lo + 1e-8)

            for method in METHOD_ORDER:
                vals = [norms[m][method] for m in radar_metrics] + [norms[radar_metrics[0]][method]]
                ax.plot(angles, vals, label=method, color=METHOD_PALETTE[method], linewidth=2)
                ax.fill(angles, vals, alpha=0.08, color=METHOD_PALETTE[method])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_metrics, fontsize=10)
            ax.set_yticklabels([])
            ax.set_title(lang, fontsize=12, fontweight="bold", pad=12)
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

        fig.suptitle(
            "RQ1 — Method comparison radar (normalised per language)\n"
            "Larger area = better overall alignment",
            fontsize=13,
        )
        plt.tight_layout()
        _save("results/rq1_radar.png")
        plt.show()

    # ------------------------------------------------------------------
    # Figure 6 — NER F1  (only shown if any non-zero values exist)
    # ------------------------------------------------------------------
    ner_df = df[df["NER_F1"].notna() & (df["NER_F1"] > 0)]
    if not ner_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=ner_df, x="lang", y="NER_F1", hue="method",
            hue_order=METHOD_ORDER, order=LANG_ORDER,
            palette=METHOD_PALETTE, ax=ax,
        )
        ax.set_title(
            "RQ1 — Extrinsic: zero-shot NER F1\n"
            "(BiLSTM-CRF trained on CoNLL-2003 → evaluated on MasakhaNER)",
            fontsize=13,
        )
        ax.set_ylim(0, max(ner_df["NER_F1"].max() * 1.4, 0.1))
        ax.set_xlabel("")
        ax.set_ylabel("Entity-level F1")
        ax.legend(title="Method")
        plt.tight_layout()
        _save("results/rq1_ner_f1.png")
        plt.show()
    else:
        print("NER F1: no non-zero values yet — skipping figure 6.")

    print("\nAll figures saved in results/")


if __name__ == "__main__":
    plot_results()
