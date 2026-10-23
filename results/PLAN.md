# Experiment plan

## Objective

Measure the effect of separate incoming and outgoing mean aggregation on directed edge recovery as the strength of directional structure varies. The encoder comparison uses a common ordered-pair decoder architecture.

## Protocol

- Five orientation strengths: 0, 0.25, 0.5, 0.75 and 1.
- Five graph and initialization seeds: 0–4.
- Four encoders: feature-only MLP, symmetrized mean aggregation, directed mean aggregation and a symmetrized model with a matched parameter budget.
- Fixed skeleton and vertex features across orientation strengths within each seed.
- Shared dyad splits and negative samples across models within each graph instance.
- Query-edge masking during training; validation and test edges excluded from message passing.
- Fixed optimization settings; checkpoint selection by validation average precision.

## Planned outputs

| Output | Purpose |
| --- | --- |
| `runs.csv` | Metrics, losses, selected epoch and timing for every run |
| `summary.csv` | Mean and sample standard deviation of test quality |
| `paired_differences.csv` | Directed-minus-symmetrized differences over paired seeds |
| `learning_curves.csv` | Training and validation loss histories |
| `paper/figures/expected_curve.pdf` | Hypothesis sketch with no measured values |
| `paper/figures/orientation_effect.pdf` | Test ROC-AUC and AP versus orientation strength |
| `paper/figures/loss_curves.pdf` | Loss curves for seed 0 at maximum orientation strength |

The initial experiment addresses a synthetic static graph. Testing an open real-world graph and additional negative-sampling schemes remains subsequent work.
