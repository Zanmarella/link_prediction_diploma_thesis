"""Mean-aggregation encoders with a common ordered-pair decoder."""

import torch
from torch import nn


def row_mean(matrix):
    return matrix / matrix.sum(1, keepdim=True).clamp_min(1)


def operators(adjacency, mode):
    if mode == "Dir":
        return [row_mean(adjacency.T), row_mean(adjacency)]
    if mode in ("Sym", "SymWide"):
        return [row_mean(((adjacency + adjacency.T) > 0).to(adjacency.dtype))]
    return []


class MeanLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_branches):
        super().__init__()
        self.self_map = nn.Linear(input_dim, output_dim)
        self.neighbor_maps = nn.ModuleList(
            nn.Linear(input_dim, output_dim, bias=False) for _ in range(n_branches))

    def forward(self, features, adjacency_operators):
        result = self.self_map(features)
        for op, layer in zip(adjacency_operators, self.neighbor_maps):
            result = result + layer(op @ features)
        return torch.relu(result)


class LinkPredictor(nn.Module):
    def __init__(self, config, mode, hidden_dim=None):
        super().__init__()
        self.mode = mode
        branches = 2 if mode == "Dir" else 0 if mode == "MLP" else 1
        hidden = config["hidden_dim"] if hidden_dim is None else hidden_dim
        emb = config["embedding_dim"]
        self.layers = nn.ModuleList([
            MeanLayer(config["feature_dim"], hidden, branches),
            MeanLayer(hidden, emb, branches)])
        self.decoder = nn.Sequential(
            nn.Linear(2 * emb, config["decoder_hidden_dim"]), nn.ReLU(),
            nn.Linear(config["decoder_hidden_dim"], 1))

    def forward(self, features, adjacency_operators, pairs):
        h = features
        for layer in self.layers:
            h = layer(h, adjacency_operators)
        ordered = torch.cat((h[pairs[:, 0]], h[pairs[:, 1]]), dim=1)
        return self.decoder(ordered).squeeze(1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def matched_width(config):
    """Choose Sym's first-layer width nearest to Dir's total parameter count."""
    d, h, e = config["feature_dim"], config["hidden_dim"], config["embedding_dim"]
    target_encoder = (3 * d + 1) * h + (3 * h + 1) * e
    return max(1, round((target_encoder - e) / (2 * d + 1 + 2 * e)))
