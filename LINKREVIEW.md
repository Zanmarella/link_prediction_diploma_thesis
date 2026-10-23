# Literature review

The review covers 40 works on directed graph representations, link prediction and evaluation. Years refer to the cited publication or the first arXiv submission. Task differences are stated explicitly.

## Directed graph representations

| Key | Work | Year | Authors | Code | Contribution | Relevance and limitations |
| --- | --- | --- | --- | --- | --- | --- |
| `dirgnn` | [Edge Directionality Improves Learning on Heterophilic Graphs](https://proceedings.mlr.press/v231/rossi24a.html) | 2024 | Rossi, Emanuele et al. | [Code](https://github.com/emalgorithm/directed-graph-neural-network) | Separates incoming and outgoing aggregation and studies directional expressivity. | Primary encoder reference. Its node-classification results motivate, but do not establish, gains in link prediction. |
| `dgcn` | [Directed Graph Convolutional Network](https://arxiv.org/abs/2004.13970) | 2020 | Tong, Zekun et al. | - | Combines first- and second-order proximity in a directed graph convolution. | Alternative directed encoder; proximity order must be controlled when comparing against a symmetrized model. |
| `digcn` | [Digraph Inception Convolutional Networks](https://proceedings.neurips.cc/paper/2020/hash/cffb6e2288a630c2a787a64ccc67097c-Abstract.html) | 2020 | Tong, Zekun et al. | [Code](https://github.com/flyingtango/DiGCN) | Uses directed diffusion and inception-style aggregation of multiple proximity orders. | Provides a multi-scale alternative; changes in diffusion and receptive field would confound a direction-only ablation. |
| `digcl` | [Directed Graph Contrastive Learning](https://proceedings.neurips.cc/paper/2021/hash/a3048e47310d6efaa4b1eaf55227bc92-Abstract.html) | 2021 | Tong, Zekun et al. | [Code](https://github.com/flyingtango/DiGCL) | Learns directed graph representations using Laplacian perturbations and a contrastive objective. | Relevant to unsupervised pretraining; the additional objective is outside the initial supervised comparison. |
| `magnet` | [MagNet: A Neural Network for Directed Graphs](https://arxiv.org/abs/2102.11391) | 2021 | Zhang, Xitong et al. | - | Encodes direction in the phase of a complex magnetic Laplacian and evaluates directed link tasks. | Spectral comparison method; introduces complex arithmetic and a charge parameter absent from the basic spatial encoder. |
| `duplex` | [DUPLEX: Dual GAT for Complex Embedding of Directed Graphs](https://arxiv.org/abs/2406.05391) | 2024 | Ke, Zhaoru et al. | [Code](https://github.com/alipay/DUPLEX) | Uses dual attention encoders for complex directed embeddings and separates connectivity from orientation. | Directly relevant to directional representation learning; requires controlling encoder and decoder changes separately. |
| `gravity` | [Gravity-Inspired Graph Autoencoders for Directed Link Prediction](https://arxiv.org/abs/1905.09570) | 2019 | Salha, Guillaume et al. | - | Introduces a gravity-inspired asymmetric decoder for graph autoencoders. | Primary directed link-prediction reference; distinguishes the contribution of the decoder from that of message passing. |
| `digae` | [Directed Graph Auto-Encoders](https://ojs.aaai.org/index.php/AAAI/article/view/20682) | 2022 | Kollias, Georgios et al. | - | Learns source and target representations with directed convolutions and an asymmetric inner-product decoder. | Close alternative combining encoder and decoder asymmetry; the two components require separate ablations. |
| `dedgat` | [DEDGAT: Dual Embedding of Directed Graph Attention Networks for Detecting Financial Risk](https://arxiv.org/abs/2303.03933) | 2023 | Wu, Jiafu et al. | - | Maintains incoming and outgoing embeddings for directed attention in financial risk classification. | Supports the use of distinct directional roles; classification results are not evidence for link-prediction performance. |
| `s2digcl` | [Toward General Digraph Contrastive Learning: A Dual Spatial Perspective](https://arxiv.org/abs/2510.16311) | 2025 | Wu, Zhengyu et al. | - | Combines complex-domain Laplacian perturbations with path-based directed contrastive views. | Recent directed representation-learning reference; contrastive augmentation must be separated from orientation retention. |
| `digtransformer` | [Transformers Meet Directed Graphs](https://arxiv.org/abs/2302.00049) | 2023 | Geisler, Simon et al. | - | Introduces magnetic-Laplacian and directional random-walk positional encodings. | Shows another way to retain direction; graph-level evaluations do not directly answer the present link-prediction question. |
| `multiq` | [What Are Good Positional Encodings for Directed Graphs?](https://arxiv.org/abs/2407.20912) | 2024 | Huang, Yinan et al. | [Code](https://github.com/Graph-COM/Multi-q-Maglap) | Defines directed walk profiles and proposes multi-q magnetic positional encodings. | Relevant to directional structural expressivity; positional encodings introduce preprocessing beyond the initial encoder comparison. |
| `msgnn` | [MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian](https://arxiv.org/abs/2209.00546) | 2022 | He, Yixuan et al. | - | Extends magnetic spectral learning to signed directed graphs and evaluates directional link tasks. | Useful spectral reference and synthetic-data design; edge signs add a factor excluded from the initial unsigned setting. |
| `sigmanet` | [SigMaNet: One Laplacian to Rule Them All](https://arxiv.org/abs/2205.13459) | 2022 | Fiorini, Stefano et al. | - | Constructs a sign-magnetic Laplacian for directed graphs with unrestricted signed weights. | Alternative treatment of direction and weights; the initial experiment fixes edge weights and does not study sign prediction. |
| `adpa` | [Breaking the Entanglement of Homophily and Heterophily in Semi-supervised Node Classification](https://arxiv.org/abs/2312.04111) | 2023 | Sun, Henan et al. | - | Studies directed versus undirected modeling and adaptively aggregates directed patterns. | Close motivation for testing when direction helps; its semi-supervised node-classification setting differs from link prediction. |

## Message-passing foundations

| Key | Work | Year | Authors | Code | Contribution | Relevance and limitations |
| --- | --- | --- | --- | --- | --- | --- |
| `mpnn` | [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) | 2017 | Gilmer, Justin et al. | - | Organizes graph learning into message, aggregation and readout operations. | Defines the notation and decomposition used to distinguish directional aggregation from edge decoding. |
| `gcn` | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | 2016 | Kipf, Thomas N. et al. | - | Uses normalized graph convolution for semi-supervised node classification. | Standard symmetrized encoder reference; a directed implementation requires an explicit normalization convention. |
| `sage` | [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) | 2017 | Hamilton, William L. et al. | - | Learns inductive node representations by sampling and aggregating neighborhoods. | Mean aggregation provides a simple base architecture for a controlled directed versus symmetrized comparison. |
| `gat` | [Graph Attention Networks](https://arxiv.org/abs/1710.10903) | 2017 | Veličković, Petar et al. | - | Learns attention weights over neighboring vertices. | Alternative aggregation mechanism; attention is an additional architectural factor beyond retaining direction. |
| `gin` | [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) | 2018 | Xu, Keyulu et al. | - | Relates aggregation functions to graph-isomorphism expressivity. | Provides theoretical context for aggregation; node-level expressivity does not by itself characterize ordered-pair prediction. |
| `gatv2` | [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) | 2021 | Brody, Shaked et al. | [Code](https://github.com/tech-srl/how_attentive_are_gats) | Identifies static-attention restrictions in GAT and introduces dynamic attention. | Controls the attention mechanism in later comparisons; it does not independently resolve directed edge modeling. |

## Link prediction

| Key | Work | Year | Authors | Code | Contribution | Relevance and limitations |
| --- | --- | --- | --- | --- | --- | --- |
| `vgae` | [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) | 2016 | Kipf, Thomas N. et al. | - | Learns latent node representations through a variational graph autoencoder. | Basic reconstruction reference; its standard inner-product decoder assigns equal scores to reversed ordered pairs. |
| `seal` | [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691) | 2018 | Zhang, Muhan et al. | - | Learns link representations from labeled enclosing subgraphs. | Strong structural alternative to node-embedding decoders; per-pair subgraph extraction increases computational cost. |
| `nbfnet` | [Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction](https://arxiv.org/abs/2106.06935) | 2021 | Zhu, Zhaocheng et al. | - | Learns pair representations through generalized Bellman-Ford path aggregation. | Captures path-dependent evidence for links; query-conditioned computation differs from a single shared node encoder. |
| `neognn` | [Neo-GNNs: Neighborhood Overlap-aware Graph Neural Networks for Link Prediction](https://arxiv.org/abs/2206.04216) | 2022 | Yun, Seongjun et al. | [Code](https://github.com/seongjunyun/Neo_GNNs) | Learns structural features based on overlapping multi-hop neighborhoods. | Motivates structural baselines; directed overlap definitions must distinguish incoming and outgoing neighborhoods. |
| `buddy` | [Graph Neural Networks for Link Prediction with Subgraph Sketching](https://arxiv.org/abs/2209.15486) | 2022 | Chamberlain, Benjamin Paul et al. | - | Approximates link-specific subgraph features using sketches and precomputation. | Provides scalable structural alternatives; their original formulation must be checked before transfer to ordered pairs. |
| `ncn` | [Neural Common Neighbor with Completion for Link Prediction](https://arxiv.org/abs/2302.00890) | 2023 | Wang, Xiyuan et al. | [Code](https://github.com/GraphPKU/NeuralCommonNeighbor) | Combines learned common-neighbor representations with completion of missing neighborhoods. | Highlights graph incompleteness and target-link handling; a directed extension requires orientation-specific common neighbors. |
| `tncn` | [Efficient Neural Common Neighbor for Temporal Graph Link Prediction](https://arxiv.org/abs/2406.07926) | 2024 | Zhang, Xiaohui et al. | [Code](https://github.com/GraphPKU/TNCN) | Uses temporal neighbor dictionaries and multi-hop common neighbors for pairwise prediction. | Recent structural link-prediction reference; event order and memory are outside the initial static experiment. |

## Network embeddings and asymmetric decoders

| Key | Work | Year | Authors | Code | Contribution | Relevance and limitations |
| --- | --- | --- | --- | --- | --- | --- |
| `node2vec` | [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) | 2016 | Grover, Aditya et al. | - | Learns node embeddings from biased random walks and evaluates link prediction. | Non-neural-message-passing reference; the walk convention and pair operator determine whether direction survives. |
| `line` | [LINE: Large-scale Information Network Embedding](https://arxiv.org/abs/1503.03578) | 2015 | Tang, Jian et al. | - | Preserves first- and second-order proximity in large network embeddings. | Relevant to source-context representations in directed networks; transductive embeddings do not use the same feature encoder as the proposed comparison. |
| `complex` | [Complex Embeddings for Simple Link Prediction](https://arxiv.org/abs/1606.06357) | 2016 | Trouillon, Théo et al. | - | Uses complex embeddings and a Hermitian scoring function to model asymmetric relations. | Decoder reference demonstrating that asymmetry can arise without directed message passing. |
| `distmult` | [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575) | 2014 | Yang, Bishan et al. | - | Uses a diagonal bilinear relation-scoring model for knowledge-base completion. | Symmetry control: for a fixed relation the score is unchanged when head and tail are exchanged. |

## Heterophily and structural controls

| Key | Work | Year | Authors | Code | Contribution | Relevance and limitations |
| --- | --- | --- | --- | --- | --- | --- |
| `h2gcn` | [Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs](https://arxiv.org/abs/2006.11468) | 2020 | Zhu, Jiong et al. | - | Separates ego and neighbor information and combines higher-order representations under heterophily. | Motivates controls for self-information and receptive field; heterophily is distinct from edge directionality. |
| `linkx` | [Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods](https://arxiv.org/abs/2110.14446) | 2021 | Lim, Derek et al. | [Code](https://github.com/CUAI/Non-Homophily-Large-Scale) | Introduces non-homophilous benchmarks and a model combining adjacency and feature representations. | Provides context for simple baselines; node classification and adjacency-row features differ from the initial link setting. |
| `critical` | [A critical look at the evaluation of GNNs under heterophily: Are we really making progress?](https://arxiv.org/abs/2302.11640) | 2023 | Platonov, Oleg et al. | [Code](https://github.com/yandex-research/heterophilous-graphs) | Identifies weaknesses and leakage in common heterophily benchmarks and introduces revised datasets. | Supports careful split and dataset validation; its conclusions concern node-classification evaluation. |

## Data and evaluation

| Key | Work | Year | Authors | Code | Contribution | Relevance and limitations |
| --- | --- | --- | --- | --- | --- | --- |
| `ogb` | [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687) | 2020 | Hu, Weihua et al. | - | Provides graph datasets with standardized task-specific splits, metrics and evaluation code. | Reference for a later real-data experiment; graph direction and the official task protocol must be checked per dataset. |
| `heart` | [Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking](https://arxiv.org/abs/2306.10453) | 2023 | Li, Juanhui et al. | [Code](https://github.com/Juanhui28/HeaRT) | Shows that baseline tuning, data splits and easy negatives distort link-prediction comparisons. | Primary evaluation reference; motivates fixed splits, common tuning budgets and a harder-negative sensitivity analysis. |
| `dynamic_eval` | [Towards Better Evaluation for Dynamic Link Prediction](https://arxiv.org/abs/2207.10128) | 2022 | Poursafaei, Farimah et al. | [Code](https://github.com/fpour/DGB) | Introduces stronger negative-sampling protocols and an edge-memorization baseline. | Demonstrates sensitivity to negative examples; temporal sampling rules should not be copied unchanged into a static task. |
| `tgb` | [Temporal Graph Benchmark for Machine Learning on Temporal Graphs](https://arxiv.org/abs/2307.01026) | 2023 | Huang, Shenyang et al. | - | Provides temporal graph benchmarks and reproducible evaluation protocols. | Defines a possible later temporal extension; chronological prediction is a different task from static edge recovery. |
| `tgb2` | [TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs](https://arxiv.org/abs/2406.09639) | 2024 | Gastinger, Julia et al. | - | Extends temporal benchmarking to knowledge graphs and heterogeneous graphs. | Clarifies the role of relation types in broader settings; multiple relation types are excluded from the initial homogeneous experiment. |

## Synthesis

Directed encoders preserve orientation through separate neighborhoods, directed diffusion or complex spectral operators. Asymmetric decoders distinguish ordered pairs even when the encoder uses a symmetrized graph. These two sources of asymmetry must therefore be evaluated separately. Results from node classification motivate the encoder design but do not establish an improvement in link prediction.

The initial comparison uses a common order-sensitive decoder with a feature-only encoder, a symmetrized mean-aggregation encoder and a Dir-GNN-style encoder. Graph generation, vertex features, train/validation/test splits, negative examples and training budgets are held fixed within each comparison. Directional structure is varied independently of graph size and expected density. ROC-AUC and average precision are reported across random seeds.

## Reference models

| Model | Role | Source |
| --- | --- | --- |
| GraphSAGE with mean aggregation on the symmetrized training graph | Basic graph encoder | `sage` |
| Dir-GNN with separate incoming and outgoing mean aggregation | Directed encoder | `dirgnn` |
| Gravity-inspired graph autoencoder | Directed decoder reference for a subsequent comparison | `gravity` |

A feature-only multilayer perceptron provides a control for the contribution of graph structure. The first experiment isolates the encoder comparison; decoder-specific alternatives are considered separately.
