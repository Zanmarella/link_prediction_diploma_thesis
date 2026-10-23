"""Checks for orientation conventions, split isolation and masked queries."""
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from data import generate
from experiment import masked_adjacency
from models import LinkPredictor, matched_width, operators, parameter_count


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parents[1] / "configs/baseline.json").read_text())

    def test_skeleton_and_features_are_fixed_across_direction_strength(self):
        a, b = generate(self.config, 0, 0), generate(self.config, 0, 1)
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.pair_split, b.pair_split)
        np.testing.assert_array_equal(a.adjacency + a.adjacency.T, b.adjacency + b.adjacency.T)
        self.assertEqual(float(np.trace(a.adjacency)), 0)

    def test_labels_dyads_and_negative_sampling(self):
        data = generate(self.config, 1, 1)
        seen = set()
        for split_id, name in enumerate(("train", "validation", "test")):
            pairs, labels = data.samples[name]
            np.testing.assert_array_equal(data.adjacency[pairs[:, 0], pairs[:, 1]], labels)
            self.assertEqual(int(labels.sum()) * 2, len(labels))
            for u, v in pairs:
                self.assertEqual(data.pair_split[u, v], split_id)
                self.assertEqual(data.pair_split[v, u], split_id)
                self.assertNotEqual(u, v)
            current = {tuple(sorted(p)) for p in pairs}
            self.assertFalse(seen & current)
            seen |= current
        np.testing.assert_array_equal(data.samples["test"][0], generate(self.config, 1, 1).samples["test"][0])

    def test_incoming_and_outgoing_operators(self):
        a = torch.zeros(3, 3)
        a[0, 1] = a[2, 1] = 1
        incoming, outgoing = operators(a, "Dir")
        torch.testing.assert_close(incoming[1], torch.tensor([.5, 0, .5]))
        torch.testing.assert_close(outgoing[0], torch.tensor([0., 1, 0]))
        torch.testing.assert_close(incoming[0], torch.zeros(3))
        torch.testing.assert_close(operators(a, "Sym")[0][1], incoming[1])

    def test_query_and_reverse_removed_without_mutating_source(self):
        a = torch.tensor([[0., 1, 0], [1, 0, 1], [0, 0, 0]])
        original = a.clone()
        masked = masked_adjacency(a, torch.tensor([[0, 1], [2, 1]]), torch.tensor([1., 0]))
        self.assertEqual(masked[0, 1].item(), 0)
        self.assertEqual(masked[1, 0].item(), 0)
        self.assertEqual(masked[1, 2].item(), 0)
        torch.testing.assert_close(a, original)

    def test_capacity_control_and_finite_gradients(self):
        directed = LinkPredictor(self.config, "Dir")
        wide = LinkPredictor(self.config, "SymWide", matched_width(self.config))
        self.assertLess(abs(parameter_count(wide) / parameter_count(directed) - 1), .01)
        data = generate(self.config, 0, .5)
        a = torch.from_numpy(data.adjacency * (data.pair_split == 0))
        for mode in self.config["models"]:
            model = LinkPredictor(self.config, mode, matched_width(self.config) if mode == "SymWide" else None)
            logits = model(torch.from_numpy(data.features), operators(a, mode), torch.tensor([[0, 1], [1, 0]]))
            self.assertEqual(tuple(logits.shape), (2,))
            logits.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
