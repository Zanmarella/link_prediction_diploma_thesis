"""Plot saved numerical results without rerunning training."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"MLP": "#777777", "Sym": "#2878b5", "Dir": "#c43c39", "SymWide": "#8a60a8"}
LABELS = {"MLP": "Feature-only MLP", "Sym": "Symmetrized", "Dir": "Directed", "SymWide": "Symmetrized (capacity control)"}


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, root, name):
    target = root / "paper/figures"
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(target / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def sketch(root):
    style()
    x = np.linspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
    ax.plot(x, 0.45 + 0.3 * x, color=COLORS["Dir"], label="Directed encoder")
    ax.plot(x, 0.45 + 0.1 * x, color=COLORS["Sym"], label="Symmetrized encoder")
    ax.set(xlabel=r"Directional structure $\rho$", ylabel="Expected prediction quality",
           yticks=[], xlim=(0, 1), title="Hypothesis sketch (not experimental measurements)")
    ax.legend(frameon=False)
    save(fig, root, "expected_curve")


def plot_all(root):
    root = Path(root)
    style()
    with (root / "results/summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, metric, title in zip(axes, ("test_auc", "test_ap"), ("ROC-AUC", "Average precision")):
        for mode in COLORS:
            selected = [r for r in rows if r["model"] == mode]
            ax.errorbar([float(r["rho"]) for r in selected],
                        [float(r[metric + "_mean"]) for r in selected],
                        yerr=[float(r[metric + "_std"]) for r in selected],
                        marker="o", markersize=4, capsize=3, color=COLORS[mode], label=LABELS[mode])
        ax.set(xlabel=r"Directional structure $\rho$", ylabel=title, xticks=[0, .25, .5, .75, 1])
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    save(fig, root, "orientation_effect")

    with (root / "results/learning_curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    config = json.loads((root / "results/metadata.json").read_text())["config"]
    seed, rho = config["seeds"][0], max(config["rho_values"])
    fig, ax = plt.subplots(figsize=(7.6, 3.4), layout="constrained")
    for mode in COLORS:
        selected = [r for r in rows if int(r["seed"]) == seed and float(r["rho"]) == rho and r["model"] == mode]
        epochs = [int(r["epoch"]) for r in selected]
        ax.plot(epochs, [float(r["validation_bce"]) for r in selected], color=COLORS[mode], label=LABELS[mode])
        ax.plot(epochs, [float(r["train_bce"]) for r in selected], color=COLORS[mode], ls="--", alpha=.65)
    ax.set(xlabel="Epoch", ylabel="Binary cross-entropy",
           title=f"Seed {seed}, rho = {rho:g}; solid: validation, dashed: training")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    save(fig, root, "loss_curves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sketch", action="store_true")
    args = parser.parse_args()
    sketch(args.root) if args.sketch else plot_all(args.root)
