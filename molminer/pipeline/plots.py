import matplotlib.pyplot as plt
import numpy as np


def plot_matches_tanimoto():
    x_labels = ["", "V1", "+ 5M", "+ Crop/Add", "V1 + 5M + Pixel Mask", "+ Crop/Add", "+ Degrade"]
    smi_data = [88.02, 90.31, 91.99, 90.09, 93.79, 92.60]
    tani_data = [0.039, 0.032, 0.036, 0.033, 0.029, 0.012]

    xticks = np.arange(1, 7)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.tick_params(axis="y", which="major", labelsize=15)
    ax1.tick_params(axis="y", which="minor", labelsize=15)
    ax1.tick_params(axis="x", labelsize=12)

    # Plot the left axis first (Exact SMILES Match)
    color = "tab:red"
    ax1.set_xlabel("Ablation Experiments", fontsize=12)
    ax1.set_ylabel("Canonical SMILES Match (%) (Higher Better)", color=color, fontsize=15)
    pl1 = ax1.plot(xticks, smi_data, "-*", color=color, label="SMILES Match")
    ax1.tick_params(axis="y", labelcolor=color)

    # Plot the right axis
    color = "tab:blue"
    ax2 = ax1.twinx()
    ax2.set_ylabel("GED (Lower Better)", color=color, fontsize=15)
    pl2 = ax2.plot(xticks, tani_data, "-^", color=color, label="Graph Edit Distance")
    ax2.tick_params(axis="y", which="major", labelcolor=color, labelsize=15)
    ax2.tick_params(axis="y", which="minor", labelcolor=color, labelsize=15)

    ax1.set_yticklabels(ax1.get_yticks(), weight="bold")
    ax1.set_xticklabels(ax1.get_xticks(), weight="bold")
    ax2.set_yticklabels(ax2.get_yticks(), weight="bold")
    ax1.set_xticklabels(x_labels)

    # Annotate the Points
    for i, label in enumerate(smi_data):
        ax1.annotate(
            str(label), (xticks[i], label), textcoords="offset points", xytext=(0, 10), ha="center"
        )
        ax2.annotate(
            str(tani_data[i]),
            (xticks[i], tani_data[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    # Create the legend
    lgs = pl1 + pl2
    lbls = [l.get_label() for l in lgs]  # noqa
    ax1.legend(lgs, lbls, loc=0)

    fig.tight_layout()
    fig.savefig("Result_Plot.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    plot_matches_tanimoto()
