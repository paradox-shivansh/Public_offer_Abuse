# Promotion Abuse Detection — Multi-Model Graph Learning

A notebook-based machine-learning project for identifying users who abuse online promotions. It combines user-level behavioural data with relationships between users, then compares three complementary classifiers:

1. **Gaussian Naive Bayes (NB)** — a lightweight tabular baseline.
2. **HistGradientBoosting (GBM)** — a non-linear model for behavioural and graph-derived features.
3. **Relation-aware GraphSAGE (GNN)** — a graph neural network that learns from each user's neighbours separately for each relationship type.

A validation-tuned weighted ensemble combines their scores.

> **Research / learning prototype.** The predictions are risk scores for investigation, not proof of fraud and not an automated basis for penalising a customer.

## 1. Problem

Promotion-abuse accounts may look ordinary when inspected individually. However, suspicious behaviour can become visible through relationships: users may share relationship identifiers, promotion-related connections, group-chat links, dealers, or other relation types. The project therefore treats the task as **node classification**:

- **Node:** a user/account.
- **Edge:** a relationship between two users.
- **Node label:** `1` = promotion abuser, `0` = normal user, `-1` internally = unlabelled.
- **Goal:** estimate the likelihood that each labelled user is an abuser.

The notebook uses the Meituan Public Promotion Abuse (PPA) dataset associated with the PromoGuardian paper. Refer to the dataset's own `readme.md` for authoritative field and relation definitions.

## 2. Project structure

The notebook expects the following layout from the project working directory:

```text
project/
├── promotion_abuse_2.ipynb
├── rasje-osfstorage-archive/
│   ├── readme.md
│   ├── node.csv
│   ├── edge.csv
│   └── Transactions/
│       ├── order_train.csv
│       └── order_test.csv
└── outputs/                  # created by the notebook
```

The dataset directory name can be changed using `DATA_DIR` in the Configuration cell. Keep the dataset files in their expected relative locations, or update the corresponding paths.

## 3. Main workflow

| Stage | What the notebook does |
|---|---|
| Dataset inventory | Lists files and prints the dataset documentation |
| Loading and schema detection | Reads node/edge CSV files and detects ID, label, endpoint and relation columns |
| ID and label processing | Unifies node identifiers, parses binary labels and retains unlabelled graph nodes |
| Graph preparation | Cleans edges, encodes relation types, removes self-loops and creates sparse adjacency matrices |
| Transaction processing | Streams weekly order files in chunks and aggregates per-user behaviour and shared-relation statistics |
| Subgraph sampling | Optionally selects a manageable graph while retaining neighbours needed for message passing |
| Data splitting | Uses a temporal split where possible, with a documented fallback |
| Feature engineering | Combines node attributes, behaviour, graph structure and training-label neighbour features |
| Model training | Fits Naive Bayes, HistGradientBoosting and relation-aware GraphSAGE |
| Ensemble | Converts model scores to validation-referenced percentile ranks and tunes vote weights on validation data |
| Evaluation | Reports PR-AUC, ROC-AUC, precision, recall, F1, lift, confusion matrix and error analysis |
| Export | Saves predictions, metrics, model artefacts and run configuration |

For the detailed explanation and mathematical formulation, read [`concept.md`](concept.md).

## 4. Environment setup

Python 3.10+ is a sensible starting point. Install the libraries used by the notebook:

```bash
uv init
uv add numpy pandas scipy matplotlib scikit-learn torch joblib ipykernel
```

If the project already has a `uv` environment, activate it and add the dependencies there. For CUDA acceleration, install a PyTorch build compatible with the installed NVIDIA driver and CUDA runtime. Use the official PyTorch installation selector to choose the correct command for your machine. The notebook automatically selects CUDA when `torch.cuda.is_available()` is true; otherwise it runs on CPU.

Register the environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name promotion-abuse --display-name "Python (promotion-abuse)"
```

Then select **Python (promotion-abuse)** in Jupyter or VS Code.

## 5. Run the notebook

1. Download and extract the PPA dataset.
2. Place `rasje-osfstorage-archive/` beside the notebook, or update `DATA_DIR`.
3. Start Jupyter from the project root.
4. Select the project kernel.
5. Run cells from top to bottom.
6. Read the detected column mapping printed by the notebook. If a mapping is incorrect, set the corresponding override in the Configuration cell.
7. Inspect the generated metrics and output files.

Do not skip the dataset documentation/inventory cells on the first run. The notebook makes assumptions about column names only as a convenience; the dataset's own documentation and the printed schema should be checked.

## 6. Important configuration

The main settings are grouped in the Configuration cell.

| Setting | Purpose |
|---|---|
| `DATA_DIR` | Dataset root folder |
| `NODE_ID_COL`, `LABEL_COL` | Optional explicit node ID and label columns |
| `EDGE_SRC_COL`, `EDGE_DST_COL`, `EDGE_REL_COL` | Optional explicit edge schema |
| `TX_USER_COL` | Optional transaction user-ID column override |
| `EXTERNAL_LABEL_FILE` | Optional separate label CSV |
| `EDGE_NROWS` | Optional edge-row limit for quick tests |
| `USE_TRANSACTIONS` | Enables transaction feature aggregation |
| `TX_MAX_ROWS` | Optional per-file row limit for trial runs |
| `TX_CHUNK` | Number of transaction rows read per chunk |
| `FORCE_RECOMPUTE_TX` | Rebuild transaction feature caches |
| `MAX_NODES` | Maximum sampled graph size; `None` requests the full graph |
| `PRESERVE_PREVALENCE` | Preserves the natural positive-class rate in sampling |
| `MAX_RELATIONS` | Maximum retained relation categories before grouping the remainder |
| `SPLIT_MODE` | `temporal` or `random` |
| `HIDDEN`, `N_LAYERS`, `DROPOUT` | GNN architecture settings |
| `LR`, `WEIGHT_DECAY`, `EPOCHS`, `PATIENCE` | GNN optimisation and early stopping |

**Recommended first test:** use a small `TX_MAX_ROWS`, a modest `EDGE_NROWS`, and a smaller `MAX_NODES` to validate paths, schemas and memory use. These limits change the data available to the model, so results from a smoke test should not be reported as final performance.

## 7. Outputs

The notebook creates an `outputs/` directory containing:

| File | Contents |
|---|---|
| `model_comparison.csv` | Validation and held-out test metrics for each model and ensemble |
| `test_predictions.csv` | Test-user IDs, model scores, graph degree and true labels |
| `gnn_best.pt` | Best validation-selected GraphSAGE model state |
| `naive_bayes.joblib` | Quantile transformer, Naive Bayes model and feature names |
| `gradient_boosting.joblib` | HistGradientBoosting model and feature names |
| `run_config.json` | Data/split settings, detected schema, GNN parameters, ensemble weights and selected model |

The notebook saves the best GNN state and the tabular model artefacts. The current export does **not** package every object needed for a one-call, standalone inference pipeline; preprocessing, graph construction and feature generation must also be reproduced consistently before deploying predictions.

## 8. Evaluation principles

Promotion abuse is a rare-event classification problem. Accuracy alone can be misleading because a model can achieve high accuracy by predicting the majority class.

The notebook therefore emphasises:

- **PR-AUC / average precision:** quality of ranking the positive class under class imbalance.
- **Precision:** fraction of flagged users that are labelled abusers.
- **Recall:** fraction of labelled abusers detected.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** general ranking quality, interpreted alongside PR-AUC.
- **Lift:** precision divided by the positive-class base rate.
- **Precision@K:** precision among the highest-scored `K` users, where `K` equals the number of positives in the test set.

The decision threshold is selected using validation data, not the test set. The final model name is also selected by validation PR-AUC. Test metrics are for the final held-out evaluation.

## 9. Limitations and responsible use

- `MAX_NODES` sampling changes the graph and can change measured performance.
- Reported abuse labels may be incomplete or noisy.
- A temporal split is only meaningful if the transaction weeks and user-activity logic represent the intended deployment setting.
- The relation-aware GNN is full-batch: large graphs can exceed memory. The notebook is not yet a neighbour-sampling implementation for millions of nodes.
- The graph representation is a simplified relation-aware GraphSAGE-style model; it is not a reproduction of every component of the published PromoGuardian method.
- Shared relationships can have legitimate explanations. A high score should trigger review, not be treated as conclusive evidence.
- Protect user identifiers and transaction information; do not publish raw predictions or identifiers without an appropriate privacy and governance review.

## 10. Further work

Potential extensions already identified in the notebook include temporal edge weights, GAT-style attention, neighbour sampling for larger graphs, and threshold selection based on the operational cost of false positives versus missed abuse. The same graph-plus-behaviour design could also be investigated for free-trial abuse using appropriate relationship signals such as shared devices or payment instruments, subject to lawful data use and privacy controls.

## References

- Meituan Public Promotion Abuse (PPA) dataset and accompanying PromoGuardian paper: [OSF project](https://osf.io/rasje/).
- PyTorch installation guidance: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
