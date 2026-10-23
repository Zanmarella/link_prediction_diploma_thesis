# A Study of Edge Orientation in Graph Neural Networks for Link Prediction

| | |
| --- | --- |
| Author | Nikita Scheglov |
| Advisor | Victor Kitov, PhD |

## Abstract

This study examines the contribution of edge orientation to link prediction with graph neural networks on directed graphs. The objective is to determine when preserving incoming and outgoing neighborhoods improves prediction of missing ordered edges compared with message passing on a symmetrized graph. The proposed comparison uses a feature-only model, a mean-aggregation graph encoder and a directed encoder with separate transformations for incoming and outgoing messages, while retaining a common order-sensitive edge decoder. A synthetic directed graph generator will control the strength of directional structure at fixed graph size and expected density, allowing the effect of orientation to be examined without simultaneously changing the amount of connectivity. Training, validation and test edges will be separated before constructing the message-passing graph, and negative examples will be drawn from absent ordered pairs under a shared sampling protocol. Prediction quality will be measured by the area under the receiver operating characteristic curve and average precision across several random seeds, with common data splits, vertex features and training budgets for all compared models. The resulting protocol is designed to distinguish improvements attributable to directional neighborhood aggregation from those attributable to decoder asymmetry or evaluation choices, and to identify conditions under which a directed encoder warrants further testing on real networks.

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

## Materials

- [Literature review](LINKREVIEW.md)
- [Code](code)
- [Paper source](paper/main.tex) · [PDF](paper/main.pdf)
- [Slides source](slides/main.tex) · [PDF](slides/main.pdf)

## License

This project is distributed under the [MIT License](LICENSE).

