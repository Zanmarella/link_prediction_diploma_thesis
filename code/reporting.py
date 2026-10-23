"""Generate LaTeX tables and numerical macros from saved measurements."""
import argparse
import csv
import json
from pathlib import Path

NAMES = {"MLP": "Feature-only MLP", "Sym": "Symmetrized",
         "Dir": "Directed", "SymWide": "Symmetrized (wide)"}


def export(root):
    root = Path(root)
    rows = list(csv.DictReader((root / "results/summary.csv").open()))
    runs = list(csv.DictReader((root / "results/runs.csv").open()))
    paired = list(csv.DictReader((root / "results/paired_differences.csv").open()))
    config = json.loads((root / "results/metadata.json").read_text())["config"]
    rho = max(config["rho_values"])
    selected = {r["model"]: r for r in rows if float(r["rho"]) == rho}
    target = root / "paper"
    target.mkdir(parents=True, exist_ok=True)
    def pm(row, metric):
        return f"${float(row[metric + '_mean']):.3f} \\pm {float(row[metric + '_std']):.3f}$"
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Test quality at $\rho=\ExperimentRhoMax$: mean and sample standard deviation over \ExperimentSeedCount\ seeds.}",
             r"\label{tab:baseline}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & Parameters & ROC-AUC & AP & Training BCE \\", r"\midrule"]
    for model, name in NAMES.items():
        row = selected[model]
        lines.append(f"{name} & {row['parameters']} & {pm(row, 'test_auc')} & {pm(row, 'test_ap')} & "
                     f"{float(row['train_bce_mean']):.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (target / "baseline_table.tex").write_text("\n".join(lines) + "\n")
    counts = [int(r['train_positive']) + int(r['validation_positive']) + int(r['test_positive']) for r in runs]
    numbers = dict(ExperimentN=str(config["n_nodes"]), ExperimentD=str(config["feature_dim"]),
                   ExperimentSeedCount=str(len(config["seeds"])), ExperimentRunCount=str(len(runs)),
                   ExperimentRhoMax=f"{rho:g}", ExperimentSeconds=f"{sum(float(r['seconds']) for r in runs):.1f}",
                   EdgeMin=str(min(counts)), EdgeMax=str(max(counts)),
                   DirParameterCount=selected["Dir"]["parameters"],
                   ControlParameterCount=selected["SymWide"]["parameters"])
    for model, prefix in (("Dir", "Directed"), ("SymWide", "Control")):
        for metric, label in (("test_auc", "AUC"), ("test_ap", "AP")):
            for stat, suffix in (("mean", "Mean"), ("std", "SD")):
                numbers[prefix + label + suffix] = f"{float(selected[model][metric + '_' + stat]):.3f}"
    for metric, label in (("test_auc", "AUC"), ("test_ap", "AP")):
        row = next(r for r in paired if float(r["rho"]) == rho and r["comparison"] == "Dir-SymWide" and r["metric"] == metric)
        numbers['Delta' + label + 'Mean'] = f"{float(row['mean']):.3f}"
        numbers['Delta' + label + 'SD'] = f"{float(row['std']):.3f}"
    (target / "experiment_numbers.tex").write_text(
        "% Generated from results/*.csv by code/reporting.py\n" +
        "\n".join("\\newcommand{\\" + name + "}{" + value + "}" for name, value in numbers.items()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    export(parser.parse_args().root)
