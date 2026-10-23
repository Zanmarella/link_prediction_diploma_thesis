# Synthetic data

The generator is implemented in [`code/data.py`](../code/data.py). Files named `synthetic_seed0_rho*.npz` contain the recorded graph instances for seed 0. All five seeds are reproduced by [`code/main.py`](../code/main.py) from the saved configuration.

| Array | Meaning |
| --- | --- |
| `features` | Vertex feature matrix, shape (180, 6) |
| `roles` | Generating group indices, used for data inspection only |
| `adjacency` | Full directed graph; entry (u, v) denotes u → v |
| `pair_split` | Unordered-pair assignment: 0 training, 1 validation, 2 test; diagonal −1 |
| `train_pairs`, `validation_pairs`, `test_pairs` | Ordered pairs in each labeled sample |
| `train_labels`, `validation_labels`, `test_labels` | Binary labels for the corresponding pairs |

The models receive `features` and the training adjacency only. The full adjacency is saved to document the generating process and labeled samples.
