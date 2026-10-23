# A Study of Edge Orientation in Graph Neural Networks for Link Prediction

| | |
| --- | --- |
| Author | Nikita Scheglov |
| Advisor | Victor Kitov, PhD |

## Abstract

This study examines how retaining edge orientation during message passing affects directed link prediction with graph neural networks. We compare a feature-only multilayer perceptron, a symmetrized mean-aggregation encoder and an encoder with separate incoming and outgoing aggregation, using the same order-sensitive decoder architecture and a further symmetrized control with a matched parameter budget. The synthetic experiment uses 180 vertices with three latent roles and noisy role-related features, while varying directional structure over five levels and retaining the same undirected skeleton and features within each of five random seeds. Unordered vertex pairs are separated into training, validation and test groups, and both orientations of each training query are masked before message passing to prevent direct edge information from entering its prediction. Across 100 model runs, checkpoints are selected by validation average precision and evaluated using test ROC-AUC and average precision under a fixed one-to-one negative sampling ratio. At maximum directional strength, the directed encoder achieves ROC-AUC 0.723 compared with 0.575 for the capacity control, with a paired mean difference of 0.148 across seeds. The advantage is absent at weak directional structure, indicating that orientation-aware aggregation is useful under specific properties of this generating family rather than uniformly beneficial, and motivating subsequent evaluation on real directed networks.

## Project description

This project studies the effect of edge orientation on link prediction with graph neural networks. The task is to estimate the probability of a directed edge between an ordered pair of vertices using the observed graph and, when available, vertex features.

The study compares message passing on the original directed graph with message passing on its symmetrized version. Incoming and outgoing neighborhoods are considered separately in the directed model.

## Objective and hypothesis

**Objective.** Assess how retaining edge orientation during message passing affects directed link prediction quality.

**Hypothesis.** Separate aggregation of incoming and outgoing neighbors improves prediction quality when edge direction carries information about vertex roles.

The comparison will use the same data splits, vertex features, training budget, and an order-sensitive edge decoder. Quality will be evaluated using ROC-AUC and average precision over several random seeds.

## Initial experimental plan

1. Construct a small synthetic directed graph with a specified distribution of edges between vertex groups.
2. Split observed edges into training, validation, and test sets. Use training edges only to construct the message-passing graph; sample negative examples from absent ordered pairs.
3. Compare a graph-independent model based on vertex features, a graph neural network with symmetrized neighborhoods, and a model with separate incoming and outgoing aggregation.
4. Vary the strength of directional structure and report prediction quality and variability across random seeds.
5. Extend the comparison to an open directed graph dataset.

## Baseline result

On the synthetic graph at maximum directional strength, the directed encoder achieves test ROC-AUC **0.723 ± 0.030**, compared with **0.575 ± 0.050** for the symmetrized control with a matched parameter budget (five seeds; mean ± sample standard deviation). The directed encoder has no consistent advantage at weak directional structure.

- [Experiment report](results/REPORT.md)
- [Numerical results](results/summary.csv)
- Reproduce: `python code/main.py`

## Materials

- [Literature review](LINKREVIEW.md)
- [Code](code)
- [Paper source](paper/main.tex) · [PDF](paper/main.pdf)
- [Slides source](slides/main.tex) · [PDF](slides/main.pdf)
- [One-minute talk](slides/talk_ru.md)

## Reproduction

```bash
python -m pip install -r code/requirements.txt
python code/main.py
```

To compile the article and one-slide PDF as part of the same run, use `python code/main.py --build-pdfs` with TeX Live, Beamer and `latexmk` installed. Plotting and LaTeX tables can also be rebuilt from saved results through `code/plot_results.py` and `code/reporting.py`.

## License

This project is distributed under the [MIT License](LICENSE).

