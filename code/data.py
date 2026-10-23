"""Synthetic directed graphs and splits of unordered vertex pairs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GraphData:
    features: np.ndarray
    roles: np.ndarray
    adjacency: np.ndarray
    pair_split: np.ndarray
    samples: dict


def generate(config, seed, rho):
    """Orient a fixed random skeleton using three cyclic vertex roles.

    A[u, v] = 1 denotes u -> v. The same seed fixes the skeleton,
    role features, orientation uniforms and pair groups across rho values.
    """
    n = config["n_nodes"]
    d = config["feature_dim"]
    q = config["edge_probability"]
    fractions = np.asarray(config["split_fractions"], dtype=float)
    if n < 6 or n % 3 or d < 3 or not 0 < q < 0.5 or not 0 <= rho <= 1:
        raise ValueError("Require n divisible by 3, d >= 3, 0 < q < .5 and 0 <= rho <= 1")
    if len(fractions) != 3 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Three positive split fractions must sum to one")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    roles = rng.permutation(np.repeat(np.arange(3), n // 3))
    features = rng.normal(0, config["feature_noise"], (n, d))
    features[np.arange(n), roles] += 1
    u, v = np.triu_indices(n, 1)
    skeleton = rng.random(len(u)) < q
    orientation_uniforms = rng.random(len(u))
    cycle = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    forward = orientation_uniforms < (1 + rho * cycle[roles[u], roles[v]]) / 2
    adjacency = np.zeros((n, n), dtype=np.float32)
    adjacency[u[skeleton & forward], v[skeleton & forward]] = 1
    adjacency[v[skeleton & ~forward], u[skeleton & ~forward]] = 1

    # Assign dyads before consulting labels; both directions share a group.
    order = rng.permutation(len(u))
    cut1 = int(fractions[0] * len(u))
    cut2 = int((fractions[0] + fractions[1]) * len(u))
    group = np.empty(len(u), dtype=np.int8)
    for split_id, indices in enumerate(np.split(order, [cut1, cut2])):
        group[indices] = split_id
    pair_split = np.full((n, n), -1, dtype=np.int8)
    pair_split[u, v] = pair_split[v, u] = group
    samples = {}
    for split_id, name in enumerate(("train", "validation", "test")):
        positive = np.argwhere((adjacency == 1) & (pair_split == split_id))
        absent = np.argwhere((adjacency == 0) & (pair_split == split_id))
        if not len(positive) or len(absent) < len(positive):
            raise ValueError("Each split requires positive edges and enough absent ordered pairs")
        neg_rng = np.random.default_rng(np.random.SeedSequence([seed, 202, split_id]))
        negative = absent[neg_rng.choice(len(absent), len(positive), replace=False)]
        pairs = np.concatenate((positive, negative))
        labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
        samples[name] = (pairs.astype(np.int64), labels.astype(np.float32))
    return GraphData(features.astype(np.float32), roles, adjacency, pair_split, samples)


def save_graph(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(features=data.features, roles=data.roles, adjacency=data.adjacency,
                  pair_split=data.pair_split)
    for name, (pairs, labels) in data.samples.items():
        arrays[name + "_pairs"] = pairs
        arrays[name + "_labels"] = labels
    np.savez_compressed(path, **arrays)
