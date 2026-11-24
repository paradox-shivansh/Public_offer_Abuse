# Concept, Knowledge Base, Workflow and Code Explanation

## 1. Project overview

This notebook builds a promotion-abuse detection pipeline using both **tabular user information** and a **multi-relational user graph**. The central hypothesis is that an abusive account may not be distinguishable from a normal account by its own attributes alone. Relationships among accounts can provide additional evidence.

The task is formulated as **binary node classification**:

- A user is represented by a graph node.
- A relationship between two users is represented by an edge.
- Edges may have different relation types.
- A labelled node has target `1` (abuser) or `0` (normal).
- A node without a known label is represented internally by `-1`. It can still participate in graph message passing.

The notebook compares three models and combines their scores:

- Gaussian Naive Bayes (NB)
- Histogram-based Gradient Boosting classifier (HistGradientBoosting / GBM)
- Relation-aware GraphSAGE-style GNN

The final weighted ensemble is selected using validation PR-AUC.

## 2. Dataset and expected files

The notebook is configured for the Meituan Public Promotion Abuse dataset (PPA), associated with the PromoGuardian paper.

Expected files:

```text
rasje-osfstorage-archive/
├── readme.md
├── node.csv
├── edge.csv
└── Transactions/
    ├── order_train.csv
    └── order_test.csv
```

### 2.1 `node.csv`

Contains user/node identifiers, a binary abuse label (or a label column that can be interpreted as binary), and possibly additional node attributes.

The notebook attempts to detect:

- Node ID column
- Label column
- Remaining node attributes

The node ID and label are removed from the ordinary attribute matrix. If labels are stored separately, `EXTERNAL_LABEL_FILE`, `EXTERNAL_LABEL_ID_COL`, and `EXTERNAL_LABEL_COL` can be configured.

### 2.2 `edge.csv`

Contains relationships between users. The notebook supports three layouts:

1. One relation column, such as `relation` or `edge_type`.
2. Multiple binary/indicator columns named like `r1`, `r2`, etc.
3. No detected relation information, in which case all edges are treated as one relation.

The source and destination endpoint columns are detected by common names or can be explicitly configured. The notebook treats edges as undirected by adding both directions to the graph representation.

### 2.3 Transaction files

The notebook expects weekly order data:

- `order_train.csv`: week 1
- `order_test.csv`: week 2

The notebook's introductory documentation describes order time, SKU, user ID and relation identifiers (`r1`–`r8`). It streams large CSVs in chunks rather than loading all order rows into memory at once.

The exact meanings of relation fields should be taken from the dataset's own `readme.md`, not inferred from their names alone.

## 3. Core concepts

### 3.1 Graph representation

A graph is written as:

\[
G=(V,E)
\]

where:

- \(V\) is the set of users.
- \(E\) is the set of user-to-user relationships.

A node can be connected to many other users. A relation-aware graph has separate edge sets:

\[
E=E_1\cup E_2\cup\cdots\cup E_R
\]

where \(R\) is the number of relation types retained by preprocessing.

The adjacency matrix for relation \(r\) is \(A_r\). Its entry is non-zero when two nodes are connected through relation \(r\). A row-normalised adjacency matrix averages neighbour information:

\[
\widetilde A_r = D_r^{-1}A_r
\]

where \(D_r\) is the diagonal matrix of row degrees. For a node with no neighbours under a relation, the corresponding sparse row contributes zero.

### 3.2 Why relation types matter

A generic edge only says that two users are connected. A typed edge preserves *how* they are connected. Different relationships may carry different behavioural meaning. The GNN therefore aggregates neighbours separately for each relation type rather than mixing all neighbours into one average.

### 3.3 Class imbalance

Promotion abuse is expected to be less common than normal activity. A model that predicts every user as normal may still have high accuracy. Therefore, precision, recall, F1 and especially PR-AUC are more informative than accuracy alone.

### 3.4 Temporal generalisation

A random split estimates performance when future examples are drawn similarly to the observed population. A temporal split is closer to a deployment scenario: train using earlier activity and test on users first appearing later. It helps reveal whether a model generalises to new users rather than simply recognising users already observed in the training period.

## 4. End-to-end workflow

```text
PPA dataset
   |
   +--> node.csv --------------------+
   |                                |
   +--> edge.csv --> clean/encode -->+--> user graph
   |                                |       |
   +--> order_train.csv ------------+       |
   +--> order_test.csv --------------> weekly user features
                                            |
                          +-----------------+------------------+
                          |                 |                  |
                    Naive Bayes       HistGradientBoosting   Relational GNN
                          |                 |                  |
                          +-----------------+------------------+
                                            |
                              validation percentile scores
                                            |
                               validation-tuned weighted vote
                                            |
                              threshold / model selection
                                            |
                             held-out test evaluation
                                            |
                        metrics, analysis and saved artefacts
```

### Stage 1 — Environment and reproducibility

The first cells import Python libraries, set display options, set random seed `42`, and check whether PyTorch and CUDA are available. The device is set to CUDA when available and otherwise CPU.

The seed improves repeatability but does not guarantee bit-for-bit identical GPU results across every environment.

### Stage 2 — Configuration

The configuration cell centralises paths, column overrides, data-size controls, split mode and model hyperparameters.

Important defaults in this notebook:

| Parameter | Default | Meaning |
|---|---:|---|
| `SEED` | 42 | Random seed |
| `USE_TRANSACTIONS` | `True` | Build weekly transaction features |
| `TX_CHUNK` | 2,000,000 | Rows read per transaction chunk |
| `MAX_NODES` | 300,000 | Target maximum sampled graph size |
| `PRESERVE_PREVALENCE` | `True` | Preserve the natural positive rate when sampling |
| `MAX_RELATIONS` | 8 | Retain up to the configured number of relation categories |
| `SPLIT_MODE` | `temporal` | Prefer time-based split |
| `VAL_FRAC_OF_TRAIN` | 0.25 | Validation share of the week-1 training pool |
| `HIDDEN` | 128 | GNN hidden width |
| `N_LAYERS` | 2 | Number of relation-aware message-passing layers |
| `DROPOUT` | 0.30 | GNN dropout |
| `LR` | 0.005 | Adam learning rate |
| `WEIGHT_DECAY` | 0.00001 | Adam weight decay |
| `EPOCHS` | 300 | Maximum GNN epochs |
| `PATIENCE` | 40 | Early-stopping patience |

`MAX_NODES=None` requests the full graph and can require substantial memory. `TX_MAX_ROWS` and `EDGE_NROWS` are intended for development runs, not final evaluation.

### Stage 3 — Dataset inventory and loading

The notebook:

1. Checks that the dataset directory exists.
2. Lists files and approximate file sizes.
3. prints the dataset's own `readme.md`.
4. Loads `node.csv` and `edge.csv`.
5. Displays shapes, memory use, sample rows and data types.

The `read_table` helper checks whether the column headers look numeric. If so, it reloads the file without a header and assigns generic column names. This is a convenience heuristic; inspect the resulting columns to make sure the file was interpreted correctly.

### Stage 4 — Schema detection

The helper `_norm` standardises column-name text. `find_col` searches likely exact names first, then common fragments.

Detected fields include:

- Node ID and label
- Edge source and destination
- Edge relation, if available
- Optional edge timestamp

If automatic detection is wrong, set the explicit override variables in the Configuration cell and rerun from the configuration/loading stage. The notebook stops if it cannot identify a label column.

### Stage 5 — ID unification and label parsing

Node IDs from the node table and both edge endpoints are converted to one common representation. The notebook creates a unified index containing IDs from all three locations.

This is important because an endpoint may appear in `edge.csv` without a corresponding row in `node.csv`. Such a user is retained as an **unlabelled graph node** rather than discarded.

The compact graph index maps each original user ID to an integer in:

\[
\{0,1,\ldots,N-1\}
\]

Labels are parsed into:

- `1`: positive / abuse
- `0`: negative / normal
- `-1`: unknown, missing or unsupported label

The text parser recognises common positive and negative words as well as numeric 0/1 values. Check the printed raw label counts if the mapping looks unexpected.

### Stage 6 — Relation encoding and edge cleaning

The notebook handles the three edge layouts described earlier. If there are many relation categories in a single relation column, it keeps frequent categories up to the configured limit and merges the remainder into `other`.

For indicator columns, a row can produce an edge for each active relation indicator. Edges with no active relation indicator can be assigned to `other`.

Self-loops are removed. The graph is treated as undirected by adding reverse edges. Duplicate edges are consolidated in the sparse adjacency representation. The resulting relation-specific adjacency matrices are used for structural features and GNN message passing.

### Stage 7 — Transaction aggregation

Large order files are processed in chunks using `aggregate_orders(path, tag, nrows=None)`. The function is designed to produce user-level summaries for each week and cache them under `outputs/` so repeated notebook runs do not have to repeat the expensive aggregation unless `FORCE_RECOMPUTE_TX=True`.

The transaction pipeline uses the user's own order activity and relation values to construct behavioural and shared-group information. These are later aligned to the unified graph node index.

The notebook documentation identifies fields such as order time, SKU, user ID and relation columns `r1`–`r8`. Exact feature definitions should be verified in the aggregation cell and against the dataset README before describing a specific feature as a domain fact.

**Temporal feature rule:** week-1 features belong to the training period; week-2 features belong to the later period. Do not accidentally combine future-period behaviour into training features when adapting the pipeline.

### Stage 8 — Graph sub-sampling

The full graph can be too large for in-memory training. When the node count exceeds `MAX_NODES`, `subsample_nodes` selects labelled seed users and includes their neighbours, then retains the induced subgraph.

Keeping neighbours is important: an unlabelled node can still provide useful context to a labelled node during message passing.

`PRESERVE_PREVALENCE=True` aims to retain the natural class prevalence in the sampled labelled population. If prevalence is not preserved, reported precision and related metrics may not represent the original dataset distribution.

After sampling, node indices, edge endpoints and transaction features are remapped to the smaller graph.

### Stage 9 — Train / validation / test split

The preferred `temporal` mode uses:

- **Training + validation pool:** labelled users active in week 1.
- **Test:** labelled users present in week 2 but not in week 1.
- **Validation:** a stratified share of the week-1 pool, controlled by `VAL_FRAC_OF_TRAIN`.

The notebook includes a fallback called `temporal_disjoint` when the strict “new users in week 2” test population is too small (the code uses a minimum-positive check). The fallback partitions users across the available temporal activity groups to obtain a usable evaluation set.

The alternative `random` mode uses stratified random train/validation/test partitions controlled by `TRAIN_FRAC` and `VAL_FRAC`. This is easier but less representative of future-user prediction.

**Important:** the actual split mode is stored in `mode` and written to `run_config.json`. Report the mode that actually ran, not merely the requested configuration.

### Stage 10 — Feature engineering

The notebook builds four feature families.

#### A. Node attributes

All usable columns in `node.csv` other than the node ID and label are considered. Categorical and numeric values are processed into a numeric representation; missing values and feature scaling are handled in the feature-building code.

#### B. Behaviour / group features

Per-user transaction summaries are aligned to graph nodes. These capture individual activity and shared-relation patterns derived from the weekly order data.

#### C. Graph-structure features

The notebook calculates graph information such as:

- Total degree
- Degree by relation
- Edge/relation multiplicity or counts
- Mean neighbour degree
- Neighbourhood averages of selected behavioural features

The neighbourhood averages use sparse matrix operations. This lets a node inherit contextual information from its connected users without explicitly looping through every neighbour in Python.

#### D. Label-aware neighbour features (GBM only)

The notebook can add counts of *known training-set* positive and negative neighbours, including relation-specific counts, plus a smoothed positive-neighbour share.

For a node \(i\), the overall positive-neighbour count can be represented as:

\[
c_i^+ = \sum_j A_{ij}\mathbb{1}(j\in T^+)
\]

and the negative-neighbour count as:

\[
c_i^- = \sum_j A_{ij}\mathbb{1}(j\in T^-)
\]

where \(T^+\) and \(T^-\) are positive and negative training nodes only.

A smoothed share is calculated in the code using the training prior:

\[
s_i = \frac{c_i^+ + \pi}{c_i^+ + c_i^- + 1}
\]

where \(\pi\) is the positive rate in the training labels.

These label-aware features are supplied to the boosting model only. Validation/test labels are not used to construct them. They are not supplied to the GNN, avoiding label information being passed through message passing.

#### Feature scaling

The raw feature matrix is retained for Naive Bayes. A `StandardScaler` is fitted on training nodes and used to prepare clipped, standardised GNN inputs. The QuantileTransformer for Naive Bayes is also fitted on training nodes only.

This train-only fitting prevents validation/test distribution information from influencing fitted preprocessing parameters.

### Stage 11 — Model A: Gaussian Naive Bayes

Naive Bayes applies Bayes' theorem with a conditional-independence assumption:

\[
P(y\mid x_1,\ldots,x_d)
\propto P(y)\prod_{k=1}^{d}P(x_k\mid y)
\]

Gaussian Naive Bayes models each feature/class conditional distribution as Gaussian:

\[
x_k\mid y=c \sim \mathcal N(\mu_{kc},\sigma_{kc}^2)
\]

The notebook first uses a `QuantileTransformer` with `output_distribution="normal"` to make heavily skewed count features more compatible with Gaussian assumptions. It is fitted only on training data.

The classifier uses `var_smoothing=1e-6`. Its output is represented as a log-odds score:

\[
s_{NB} = \log P(y=1\mid x)-\log P(y=0\mid x)
\]

**Role:** a fast baseline that can work with relatively little positive data. Its limitation is that it does not learn graph message passing; graph information only reaches it through engineered features.

### Stage 12 — Model C: HistGradientBoosting

The tabular boosting model is `HistGradientBoostingClassifier`. Gradient boosting builds an additive collection of decision trees, where later trees correct residual errors made by earlier trees.

The configured model uses:

- `learning_rate=0.06`
- `max_iter=800`
- `max_leaf_nodes=31`
- `min_samples_leaf=40`
- `l2_regularization=1.0`
- `class_weight="balanced"`
- early stopping

It receives the engineered feature matrix plus optional label-aware training-neighbour features. Its decision function supplies the model score.

**Role:** learn non-linear feature interactions and behavioural patterns from tabular and engineered graph features.

### Stage 13 — Model B: relation-aware GraphSAGE

The GNN is implemented directly in PyTorch; it does not require PyTorch Geometric or DGL.

For node \(i\) at layer \(l\), the model concatenates:

- Its current representation \(h_i^{(l)}\)
- The mean representation of neighbours under relation 1
- The mean representation of neighbours under relation 2
- …
- The mean representation of neighbours under relation \(R\)

The layer update is:

\[
h_i^{(l+1)} =
\sigma\left(
W^{(l)}
\left[
h_i^{(l)}
\;\|\;
\operatorname{mean}_{j\in N_1(i)}h_j^{(l)}
\;\|\;\cdots\|\;
\operatorname{mean}_{j\in N_R(i)}h_j^{(l)}
\right]
\right)
\]

Here:

- \(N_r(i)\) is the set of neighbours of node \(i\) through relation \(r\).
- \(\|\) means concatenation.
- \(W^{(l)}\) is a learned linear transformation.
- \(\sigma\) is ReLU on hidden layers.
- The final layer produces one logit per node.

Each relation's adjacency matrix is stored as a sparse PyTorch tensor. Neighbour aggregation uses `torch.sparse.mm`, which avoids converting the graph into a dense matrix.

Training details:

- Optimiser: Adam
- Loss: `BCEWithLogitsLoss`
- Positive class weighting: based on the training class ratio, clipped to a maximum of 50
- Dropout: configured by `DROPOUT`
- Early stopping: validation PR-AUC with `PATIENCE`
- Best model state: the epoch with the highest validation PR-AUC

Unlabelled nodes are assigned a placeholder target only to create a tensor of the right shape; the loss is calculated on `tr_idx` only. They therefore contribute graph context without contributing supervised loss.

**Full-batch limitation:** each epoch performs message passing over the complete sampled graph. This is convenient for a moderate graph but may not fit a graph with millions of nodes. Neighbour sampling or mini-batch graph training would be needed for that scale.

### Stage 14 — Score normalisation and ensemble

The models output scores on different scales. The notebook converts each model's scores to percentile ranks using the validation scores as the reference distribution.

For reference scores \(S_{val}\), the empirical percentile of a score \(s\) is:

\[
r(s)=\frac{1}{|S_{val}|}\sum_{v\in S_{val}}\mathbb{1}(v\leq s)
\]

The weighted ensemble is:

\[
S_{ens}=w_{GNN}r_{GNN}+w_{GBM}r_{GBM}+w_{NB}r_{NB}
\]

with:

\[
w_{GNN}+w_{GBM}+w_{NB}=1,\qquad w_k\geq 0
\]

The notebook searches a grid of weights in increments of 0.1 and chooses the combination with the best validation PR-AUC.

It also trains an optional stacked logistic-regression model using the three model scores and validation labels. Because the stacker is fitted on the validation set, its validation performance is not an unbiased estimate; the held-out test set remains the final evaluation.

### Stage 15 — Threshold selection and evaluation

For each model score, `best_threshold` selects the validation threshold that maximises F1. That threshold is then applied to test scores.

The notebook reports:

| Metric | Interpretation |
|---|---|
| Precision | Of flagged users, fraction labelled abuse |
| Recall | Of labelled abusers, fraction detected |
| F1 | Harmonic mean of precision and recall |
| PR-AUC | Area under precision-recall curve / average precision |
| ROC-AUC | Area under ROC curve |
| Accuracy | Overall fraction correct; potentially misleading under imbalance |
| Balanced accuracy | Mean of sensitivity and specificity |
| Lift | Precision divided by test positive-class prevalence |
| Precision@K | Precision among top K scores, with K equal to the number of test positives |
| Flagged | Number of test users above the validation threshold |

The notebook also plots:

- Test precision-recall curves
- Confusion matrix for the selected model
- Score distributions for normal and abusive test users
- Feature effect sizes
- Permutation importance for the boosting model

The final model is selected by **validation PR-AUC**, not by test performance. The test set is intended to remain a held-out report card.

### Stage 16 — Error analysis

The notebook examines:

- Standardised mean differences between positive and negative training users
- Permutation importance on a validation sample
- Highest-scoring test users
- Missed positive users and their graph degree

The notebook notes that users with few connections may be harder for a graph model to identify because they provide less neighbourhood context. This is a diagnostic observation, not a guarantee about every missed case.

### Stage 17 — Saving outputs

The notebook writes:

- `outputs/model_comparison.csv`
- `outputs/test_predictions.csv`
- `outputs/gnn_best.pt`
- `outputs/naive_bayes.joblib`
- `outputs/gradient_boosting.joblib`
- `outputs/run_config.json`

Transaction aggregation also uses cached feature files in `outputs/` so the large CSV files need not be reprocessed on every run.

The run configuration records the actual split mode, graph size, feature count, relation names, detected columns, sampling settings, GNN parameters, ensemble weights and selected model.

## 5. Leakage prevention — what is protected and what to verify

Leakage occurs when information unavailable at prediction time influences training or evaluation.

The notebook includes several controls:

1. The temporal split aims to keep week-2-only users in the test set.
2. Users active in both weeks are excluded from the strict new-user test group.
3. Scalers and the Naive Bayes quantile transformer are fitted on training nodes.
4. Label-aware neighbour features use training labels only.
5. Label-aware neighbour features are not fed to the GNN.
6. The GNN's supervised loss is calculated on training indices only.
7. Ensemble weights and classification thresholds are selected on validation data.
8. The final model choice is based on validation PR-AUC.

**Important audit point:** graph topology and feature timing still need to match the intended real-world prediction moment. If the graph includes relationships that would only be known after the prediction date, or if week-2 behavioural aggregates are made available to a model that is meant to predict before week 2, the evaluation may still be optimistic. Confirm the exact timing semantics in the dataset and aggregation code before making deployment claims.

## 6. Configuration guidance and practical run plan

### First smoke test

Use small data limits to validate:

- Dataset paths
- CSV parsing
- Detected columns
- Label values
- Relation encoding
- Transaction user-ID detection
- Memory consumption

A smoke test is a pipeline check only; do not present its metrics as final.

### Full experiment

After the smoke test:

1. Remove row limits where resources allow.
2. Decide whether the graph can fit in memory.
3. Set `MAX_NODES` to a justified sample size or `None` for full graph.
4. Keep `PRESERVE_PREVALENCE=True` when measuring realistic precision.
5. Use the temporal split if the data supports it.
6. Save the actual configuration with the results.
7. Repeat runs if you need uncertainty estimates; one random seed is not a stability study.

### CUDA

CUDA can accelerate the PyTorch GNN portion. It does not automatically move pandas, SciPy, scikit-learn or CSV aggregation to the GPU. The notebook's sparse adjacency tensors and GNN tensors are moved to the selected PyTorch device. Memory can still be a limiting factor.

## 7. Interpretation and responsible use

A high prediction score means the model found patterns associated with labelled abuse in this dataset. It does not prove that a particular user intentionally committed fraud.

Potential sources of error include:

- Incorrect or incomplete labels
- Shared relationships that have legitimate explanations
- Users with few graph connections
- Distribution changes between the two weeks
- Subgraph sampling effects
- Missing or malformed transaction records
- Schema auto-detection mistakes

Use model scores to prioritise human review. Before operational use, define an appeal process, measure false-positive impact, assess privacy and legal requirements, and validate across time periods and user groups where appropriate.

## 8. Current limitations and future improvements

The notebook itself identifies these limitations and extensions:

- Sampling to `MAX_NODES` thins the graph and changes the experiment.
- Reported abuse labels may contain false negatives.
- Full-batch GNN training does not scale to millions of nodes without neighbour sampling.
- The model is a simplified relation-aware GraphSAGE implementation, not a complete reproduction of the published PromoGuardian system.
- Temporal edge weights could represent changing relationships.
- GAT-style attention could learn which neighbours matter most.
- Neighbour sampling could support larger graphs.
- Thresholds could be selected using the business cost of wrongly blocking a legitimate customer versus missing an abusive account.

For a free-trial-abuse adaptation, the same general pattern may be investigated with carefully governed relationship signals such as shared devices, IP ranges, payment instruments or repeated account-creation patterns. Such signals can also connect legitimate users, so they require cautious interpretation and privacy review.

## 9. Suggested viva / project explanation

**Why use a graph?** Because accounts may look normal individually, while shared relationships reveal coordinated behaviour.

**Why use relation-aware aggregation?** Different edge types can have different meanings. Separate aggregation lets the model learn from each type independently.

**Why keep unlabelled nodes?** They can still provide neighbourhood information even though they do not contribute to supervised loss.

**Why Naive Bayes?** It is a fast, simple baseline and gives a useful comparison against more complex models.

**Why gradient boosting?** It learns non-linear interactions in behavioural and engineered graph features.

**Why a GNN?** It learns representations by combining a user's own features with information from connected users.

**Why an ensemble?** The models use different signals and assumptions. Combining their validation-calibrated ranks may improve ranking quality if their errors are complementary. The experiment must demonstrate that; an ensemble is not automatically better.

**Why PR-AUC?** It is informative when the positive class is rare and focuses on the precision-recall trade-off.

**Why temporal split?** It better approximates predicting future users from earlier data than a random split.

**How is leakage reduced?** Train-only preprocessing, train-label-only neighbour features, training-only GNN loss, and validation-only model/threshold selection.

**What is the main limitation?** The full-batch GNN and in-memory graph operations limit scale; sampling changes the graph, and reported labels may be incomplete.

## 10. References and source of truth

- Dataset: Meituan Public Promotion Abuse (PPA), associated with PromoGuardian: https://osf.io/rasje/
- The dataset's own `readme.md` is the source of truth for CSV field definitions and relation semantics.
- The notebook `promotion_abuse_2.ipynb` is the source of truth for the exact implementation, configuration defaults and exported artefacts.

This document explains the notebook as uploaded. It does not claim that the code has been executed successfully on the full dataset or that any particular performance score has been achieved.
