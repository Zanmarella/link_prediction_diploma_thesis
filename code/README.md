# Synthetic directed link prediction

## Installation

Python 3.12 was used for the recorded experiment. From the repository root:

```bash
python -m pip install -r code/requirements.txt
```

## Reproduction

```bash
python code/main.py
```

The command generates the data, trains the models, saves numerical results and builds the figures. Configuration is stored in [configs/baseline.json](configs/baseline.json). Runs use deterministic CPU operations with one thread. Execution times and platform metadata can differ across machines.

To rebuild figures from saved results:

```bash
python code/plot_results.py
```

To draw the pre-experiment hypothesis sketch:

```bash
python code/plot_results.py --sketch
```

To run protocol checks:

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -v
```

## Data

Vertices have three equally sized roles. An undirected Erdos-Renyi skeleton is sampled with edge probability 0.15 and each skeleton edge receives exactly one direction. For roles a and b, the probability of u → v is (1 + ρ C[a,b])/2, where C[0,1] = C[1,2] = C[2,0] = 1, C is skew-symmetric and its diagonal is zero. Thus ρ = 0 gives random orientation and ρ = 1 gives cyclic orientation between different roles. The skeleton and total edge count remain identical across ρ for a given seed.

Features are six-dimensional Gaussian noise with standard deviation 1; one is added to the coordinate corresponding to the role among the first three coordinates. True roles are used only for graph generation and are not passed to the models. Saved arrays for seed 0 are stored in `data/`; other instances are reconstructed from the configuration.

Unordered vertex pairs are assigned to training, validation and test groups in proportions 80/10/10. Negatives are sampled from absent ordered pairs in the corresponding group at a 1:1 ratio. All models use identical samples within a graph instance. Both orientations of every training query pair are removed before message passing, including the reverse edge of a negative query. Evaluation uses training edges only.

## Models and optimization

All models use two ReLU encoder layers, a 32-dimensional embedding and the same ordered-concatenation decoder architecture (64 → 32 → 1). Decoder weights are fitted separately.

- `MLP`: vertex features only; hidden width 32.
- `Sym`: self and symmetrized-neighborhood mean transformations; hidden width 32.
- `Dir`: self, incoming-mean and outgoing-mean transformations; hidden width 32.
- `SymWide`: symmetrized encoder with hidden width 48, chosen to match `Dir`'s parameter count within 1%.

Training uses Adam with learning rate 0.01, mean binary cross-entropy and λ = 10⁻⁴ times the sum of squared parameters. Batches of up to 512 pairs are fixed within each run. The maximum is 120 epochs, with early stopping after 25 epochs without improvement in validation average precision. The checkpoint with the highest validation average precision is evaluated on test pairs. No hyperparameter search was performed.

The encoder designs follow [GraphSAGE](https://arxiv.org/abs/1706.02216) and [Dir-GNN](https://proceedings.mlr.press/v231/rossi24a.html). This is a minimal implementation for a controlled synthetic experiment.

## Outputs

- `results/runs.csv`: metrics, selected epochs, parameter counts and timings for each run.
- `results/summary.csv`: means and sample standard deviations over five seeds.
- `results/paired_differences.csv`: paired differences between directed and symmetrized models.
- `results/learning_curves.csv`: training and validation losses and validation AP by epoch.
- `results/metadata.json`: configuration, software versions and source hashes.
- `paper/figures/`: the orientation comparison, loss curves and hypothesis sketch.

Training curves record mean batch BCE during an epoch. The `train_bce` field in `runs.csv` is recomputed at the selected checkpoint with query masking; `train_objective` also includes the L2 penalty. Error bars in the comparison figure show sample standard deviations, not confidence intervals.
