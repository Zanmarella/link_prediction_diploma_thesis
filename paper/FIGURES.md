# Figures

All measured figures read saved data or CSV outputs through `code/plot_results.py`. They can be rebuilt without training.

| File in `figures/` | Content | Use |
| --- | --- | --- |
| `data_example.pdf` | Training adjacency before and after symmetrization, sorted by generating roles | Data description in the article |
| `orientation_effect.pdf` | Test ROC-AUC and average precision over all encoders and orientation strengths | Complete experiment comparison in the article |
| `loss_curves.pdf` | Training and validation BCE at seed 0 and maximum directional strength | Optimization behavior in the article |
| `one_slide_chart.pdf` | Directed encoder versus the capacity control, with test ROC-AUC and sample SD | Main message of the one-slide talk |
| `expected_curve.pdf` | Schematic hypothesis without measured values | Experiment plan |

PNG versions support previews in Markdown. `expected_curve.pdf` is generated separately with `python code/plot_results.py --sketch`.
