"""Train and evaluate models; numerical outputs contain no plotting code."""

import copy
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from data import generate, save_graph
from models import LinkPredictor, matched_width, operators, parameter_count


def masked_adjacency(adjacency, pairs, labels):
    result = adjacency.clone()
    # Mask both orientations for every query, including negative queries
    # whose reverse orientation may be a positive training edge.
    result[pairs[:, 0], pairs[:, 1]] = 0
    result[pairs[:, 1], pairs[:, 0]] = 0
    return result


def make_batches(pairs, labels, seed, size):
    order = np.random.default_rng(np.random.SeedSequence([seed, 303])).permutation(len(pairs))
    return [(pairs[idx], labels[idx]) for idx in
            (order[i:i + size] for i in range(0, len(order), size))]


@torch.no_grad()
def evaluate(model, features, ops, pairs, labels):
    model.eval()
    logits = model(features, ops, pairs)
    probabilities = torch.sigmoid(logits).numpy()
    return dict(bce=float(nn.functional.binary_cross_entropy_with_logits(logits, labels)),
                auc=float(roc_auc_score(labels.numpy(), probabilities)),
                ap=float(average_precision_score(labels.numpy(), probabilities)))


def fit(config, data, seed, mode):
    torch.manual_seed(seed + 1000)
    model = LinkPredictor(config, mode, matched_width(config) if mode == "SymWide" else None)
    x = torch.from_numpy(data.features)
    adjacency = torch.from_numpy(data.adjacency * (data.pair_split == 0))
    samples = {k: (torch.from_numpy(p), torch.from_numpy(y)) for k, (p, y) in data.samples.items()}
    batches = make_batches(*samples["train"], seed, config["batch_size"])
    batch_ops = [operators(masked_adjacency(adjacency, p, y), mode) for p, y in batches]
    full_ops = operators(adjacency, mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    best_ap, best_epoch, best_state = -1, 0, None
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_bce = 0
        for (pairs, labels), ops in zip(batches, batch_ops):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, ops, pairs)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            penalty = sum(p.square().sum() for p in model.parameters())
            loss = bce + config["l2_lambda"] * penalty
            loss.backward()
            optimizer.step()
            train_bce += float(bce.detach()) * len(labels)
        validation = evaluate(model, x, full_ops, *samples["validation"])
        history.append(dict(epoch=epoch, train_bce=train_bce / len(samples["train"][1]),
                            validation_bce=validation["bce"], validation_ap=validation["ap"]))
        if validation["ap"] > best_ap + 1e-10:
            best_ap, best_epoch = validation["ap"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config["patience"]:
            break
    model.load_state_dict(best_state)
    validation = evaluate(model, x, full_ops, *samples["validation"])
    test = evaluate(model, x, full_ops, *samples["test"])
    # Evaluate the internal criterion at the selected checkpoint using the same masks.
    with torch.no_grad():
        train_bce = sum(float(nn.functional.binary_cross_entropy_with_logits(
            model(x, ops, pairs), labels)) * len(labels)
            for (pairs, labels), ops in zip(batches, batch_ops)) / len(samples["train"][1])
        l2 = float(sum(p.square().sum() for p in model.parameters()))
    result = dict(model=mode, seed=seed, parameters=parameter_count(model),
                  best_epoch=best_epoch, epochs_run=epoch, train_bce=train_bce,
                  train_objective=train_bce + config["l2_lambda"] * l2,
                  validation_bce=validation["bce"], validation_ap=validation["ap"],
                  test_bce=test["bce"], test_auc=test["auc"], test_ap=test["ap"],
                  seconds=time.perf_counter() - started,
                  train_positive=int(data.samples["train"][1].sum()),
                  validation_positive=int(data.samples["validation"][1].sum()),
                  test_positive=int(data.samples["test"][1].sum()))
    return result, history


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for rho in sorted({r["rho"] for r in rows}):
        for model in dict.fromkeys(r["model"] for r in rows):
            group = [r for r in rows if r["rho"] == rho and r["model"] == model]
            row = dict(rho=rho, model=model, n=len(group), parameters=group[0]["parameters"])
            for metric in ("test_auc", "test_ap", "train_bce", "test_bce", "seconds"):
                values = [r[metric] for r in group]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary.append(row)
    return summary


def run(config, root):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    rows, curves = [], []
    for seed in config["seeds"]:
        for rho in config["rho_values"]:
            data = generate(config, seed, rho)
            if seed == config["seeds"][0]:
                save_graph(data, root / "data" / f"synthetic_seed{seed}_rho{rho:.2f}.npz")
            for mode in config["models"]:
                result, history = fit(config, data, seed, mode)
                result = dict(rho=rho, **result)
                rows.append(result)
                curves.extend(dict(rho=rho, seed=seed, model=mode, **r) for r in history)
                print(f"seed={seed} rho={rho:.2f} {mode}: AUC={result['test_auc']:.4f} "
                      f"AP={result['test_ap']:.4f} epoch={result['best_epoch']}", flush=True)
                write_csv(result_dir / "runs.csv", rows)
                write_csv(result_dir / "learning_curves.csv", curves)
    summary = summarize(rows)
    write_csv(result_dir / "summary.csv", summary)
    paired = []
    for rho in config["rho_values"]:
        for control in ("Sym", "SymWide"):
            for metric in ("test_auc", "test_ap"):
                differences = []
                for seed in config["seeds"]:
                    group = {r["model"]: r for r in rows if r["rho"] == rho and r["seed"] == seed}
                    differences.append(group["Dir"][metric] - group[control][metric])
                paired.append(dict(rho=rho, comparison="Dir-" + control, metric=metric,
                                   mean=float(np.mean(differences)),
                                   std=float(np.std(differences, ddof=1)) if len(differences) > 1 else 0))
    write_csv(result_dir / "paired_differences.csv", paired)
    metadata = dict(config=config, python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, torch=torch.__version__, sklearn=sklearn.__version__,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path(__file__).parent.glob("*.py"))})
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary
