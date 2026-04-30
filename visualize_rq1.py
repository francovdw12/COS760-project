import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(csv_path: str = "results/rq1_results.csv"):
    df = pd.read_csv(csv_path)
    metrics = ["P@1", "P@5", "MCS", "CKA_after"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=df, x="language", y=metric,
            hue="method", palette="Set2", ax=ax
        )
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.legend(title="Method")

    plt.suptitle("RQ1 - Comparison of alignment methods\n"
                 "(isiZulu=zul, Sepedi=nso, Setswana=tsn)", y=1.02)
    plt.tight_layout()
    plt.savefig("results/rq1_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()