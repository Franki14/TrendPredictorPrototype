# TrendPredictorPrototype

## Predictive Modelling of Social Media Trend Emergence

TrendPredictorPrototype is a data-driven prototype for predicting whether an early social-media information cascade will develop into an emerging trend.

The project investigates the research question:

> **Can a data-driven model predict the emergence and spread of trends on social media platforms?**

The system combines **temporal cascade characteristics** with **network diffusion characteristics** and compares interpretable machine-learning approaches for early trend-emergence prediction.

The final prototype includes:

- cascade preprocessing and feature construction;
- temporal and network diffusion features;
- baseline classifiers;
- Logistic Regression and Random Forest models;
- class-imbalance experiments;
- feature-group ablation;
- hyperparameter experiments;
- bootstrap confidence intervals and statistical model comparison;
- model explainability and partial-dependence analysis;
- a frozen final Random Forest model;
- a Viral Potential Score (VPS);
- a Streamlit dashboard for exploring historical cascades and model predictions;
- end-to-end inference tests.

---

## 1. Prediction Task

The project treats trend emergence as a **binary classification problem**.

Each cascade is represented using information available from its early development. The target variable is:

- `0` — Non-Emerging Trend
- `1` — Emerging Trend

The processed data is separated into training, validation and held-out test sets.

The validation set is used for model comparison, feature experiments and model-selection decisions. The held-out test set is reserved for final evaluation of the frozen model.

---

## 2. Final Predictive Features

The final model uses eight features covering temporal behaviour and network diffusion.

### Temporal features

| Feature | Description |
| --- | --- |
| `TimeTo5` | Time required for the cascade to reach its early observation size |
| `MinInterarrival` | Minimum time between early cascade events |
| `EarlyAcceleration` | Change in the early rate of cascade growth |

### Network features

| Feature | Description |
| --- | --- |
| `NeighbourhoodReach` | Reach of the cascade through the surrounding network |
| `EarlyDensity` | Density of connections within the early cascade |
| `MeanClustering` | Mean clustering coefficient of early participants |
| `CommunityDiversity` | Diversity of network communities reached by the cascade |
| `MeanPageRank` | Mean PageRank centrality of early participants |

Outcome-related and retrospective fields such as `Target` and `FinalSize` are not used as model inputs.

---

## 3. Models

The main predictive models are:

- Balanced Logistic Regression
- Balanced Random Forest

Additional experiments include:

- most-frequent baseline;
- stratified-random baseline;
- unweighted versus class-balanced Random Forest;
- temporal-only features;
- network-only features;
- combined temporal and network features;
- Random Forest hyperparameter experiments.

The final frozen model is:

```text
Model: Balanced Random Forest
Trees: 500
Class weight: balanced
Random state: 42
Classification threshold: 0.50
Number of features: 8
```

The frozen configuration is stored in:

```text
models/model_metadata.json
```

---

## 4. Validation Results

The principal validation-set comparison was:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Most Frequent Baseline | 0.7067 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2933 |
| Stratified Random Baseline | 0.5618 | 0.2588 | 0.2651 | 0.2619 | 0.4750 | 0.2842 |
| Balanced Logistic Regression | 0.5477 | 0.3869 | 0.9277 | 0.5461 | 0.6513 | 0.3789 |
| Balanced Random Forest | 0.7809 | 0.5913 | 0.8193 | 0.6869 | 0.8584 | 0.7287 |

The Balanced Logistic Regression produced very high recall but substantially lower precision and discrimination.

The Balanced Random Forest provided a stronger balance across F1, ROC-AUC and PR-AUC while retaining relatively high recall.

---

## 5. Feature-Group Ablation

The contribution of temporal and network information was evaluated separately.

| Feature Group | Accuracy | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Temporal Only | 0.5265 | 0.3679 | 0.5402 | 0.3377 |
| Network Only | 0.7456 | 0.5714 | 0.7758 | 0.5909 |
| Combined | 0.7809 | 0.6869 | 0.8584 | 0.7287 |

Network features provided substantially more predictive information than the temporal-only feature set in this dataset.

The strongest results were obtained when temporal and network characteristics were combined, supporting the project's multidimensional approach to modelling trend emergence.

---

## 6. Statistical Evaluation

Bootstrap resampling was used to quantify uncertainty around validation metrics.

For the Balanced Random Forest:

| Metric | Estimate | 95% Bootstrap CI |
| --- | ---: | ---: |
| Accuracy | 0.7809 | [0.7314, 0.8269] |
| Precision | 0.5913 | [0.5000, 0.6783] |
| Recall | 0.8193 | [0.7333, 0.8989] |
| F1 | 0.6869 | [0.6070, 0.7562] |
| ROC-AUC | 0.8584 | [0.8122, 0.9019] |
| PR-AUC | 0.7287 | [0.6401, 0.8107] |

A paired bootstrap comparison between the Balanced Random Forest and Balanced Logistic Regression found higher Random Forest estimates for accuracy, precision, F1, ROC-AUC and PR-AUC, while Logistic Regression had higher recall.

McNemar's test also identified a statistically significant difference in their paired classification errors.

Detailed outputs are stored under:

```text
results/tables/
```

---

## 7. Held-Out Test Evaluation

After model-selection decisions were completed using validation evidence, the final Balanced Random Forest configuration was evaluated on the held-out test set.

| Metric | Test Estimate | 95% Bootstrap CI |
| --- | ---: | ---: |
| Accuracy | 0.7912 | [0.7441, 0.8384] |
| Precision | 0.4643 | [0.3611, 0.5699] |
| Recall | 0.6964 | [0.5692, 0.8182] |
| F1 | 0.5571 | [0.4545, 0.6490] |
| ROC-AUC | 0.8257 | [0.7598, 0.8865] |
| PR-AUC | 0.6084 | [0.4784, 0.7261] |

The held-out confusion matrix was:

```text
[[196  45]
 [ 17  39]]
```

The test results are treated as the final estimate of the frozen model's performance rather than as an additional model-selection stage.

---

## 8. Model Explainability

Random Forest behaviour is investigated using:

- impurity-based feature importance;
- permutation importance;
- one-way partial dependence;
- two-way partial dependence;
- feature-group ablation.

Permutation importance identified `NeighbourhoodReach` and `EarlyDensity` as particularly influential validation features, followed by temporal and additional network characteristics.

The difference between impurity and permutation rankings is retained as part of the analysis rather than assuming a single importance method gives a complete explanation of model behaviour.

Explainability figures are stored in:

```text
results/figures/
```

---

## 9. Viral Potential Score

The prototype exposes the model output through a **Viral Potential Score (VPS)**.

The score is defined as:

```text
VPS = P(Emerging Trend | model features) × 100
```

For example, a model probability of:

```text
0.708
```

produces:

```text
VPS = 70.8 / 100
```

VPS is therefore a user-facing transformation of the Random Forest's predicted positive-class probability.

It should **not** be interpreted as an independently validated real-world probability that a topic will become viral. Its meaning is conditional on the trained model, dataset, target definition and selected features.

---

## 10. Dashboard

The Streamlit prototype provides three principal views:

### Trend Predictions

Displays the selected cascade's:

- Viral Potential Score;
- predicted outcome;
- model probability;
- historical outcome;
- final cascade size;
- model input features.

### Trend Explorer

Allows historical cascades to be inspected using temporal and network characteristics alongside their model prediction and historical outcome.

### Feature Analysis

Presents experimental and explainability outputs, including:

- permutation importance;
- partial dependence;
- network-feature interaction analysis;
- feature-group ablation.

The dashboard is intended as an exploratory research prototype rather than a production forecasting service.

---

## 11. Repository Structure

```text
TrendPredictorPrototype/
│
├── app.py
├── README.md
├── requirements.txt
├── socmed_environment.yml
│
├── assets/
│   └── style.css
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── model_metadata.json
│   └── final_random_forest.joblib
│
├── results/
│   ├── data_hashes.txt
│   ├── figures/
│   └── tables/
│
└── src/
    ├── baseline_models.py
    ├── cascade_preprocess.py
    ├── dashboard_data.py
    ├── feature_engineering.py
    ├── feature_selection.py
    ├── final_test_evaluation.py
    ├── final_test_visualisations.py
    ├── logistic_regression.py
    ├── model_explainability.py
    ├── network_analysis.py
    ├── network_features.py
    ├── network_statistics.py
    ├── predict.py
    ├── preprocess.py
    ├── random_forest.py
    ├── results_summary.py
    ├── results_visualisation.py
    ├── statistical_evaluation.py
    ├── test_end_to_end.py
    ├── test_prediction_pipeline.py
    └── train_final_model.py
```

Raw and generated data files may be excluded from version control according to `.gitignore`.

The trained `.joblib` model is also excluded from Git and can be regenerated from the frozen training data.

---

## 12. Installation

Python 3.11 is recommended.

Create and activate a virtual environment if desired, then install the project dependencies:

```bash
pip install -r requirements.txt
```

NLTK's VADER sentiment analyser requires the VADER lexicon. Install it once with:

```bash
python -m nltk.downloader vader_lexicon
```

A fuller record of the development environment is retained in:

```text
socmed_environment.yml
```

---

## 13. Data and Feature Pipeline

The project contains separate scripts for different stages of the experimental pipeline.

Raw cascade/network processing includes:

```bash
python src/cascade_preprocess.py
python src/network_features.py
```

Earlier text-oriented preprocessing and sentiment feature engineering are contained in:

```bash
python src/preprocess.py
python src/feature_engineering.py
```

The final predictive experiments use the network-enriched cascade datasets:

```text
data/processed/cascade_train_network.csv
data/processed/cascade_validation_network.csv
data/processed/cascade_test_network.csv
```

Because generated processed datasets and raw data may be excluded from version control, reproducing them requires access to the corresponding source data.

---

## 14. Rebuilding the Final Model

The final model artifact is intentionally not tracked in Git because it is a generated binary file.

With the frozen training dataset available, rebuild it using:

```bash
python src/train_final_model.py
```

This creates:

```text
models/final_random_forest.joblib
models/model_metadata.json
```

The metadata records the model configuration, feature order, training-row count and SHA-256 hash of the frozen training dataset.

---

## 15. Running the Prediction Pipeline

A basic prediction-pipeline demonstration can be run with:

```bash
python src/predict.py
```

Pipeline consistency can be checked using:

```bash
python src/test_prediction_pipeline.py
```

The full end-to-end system test is:

```bash
python src/test_end_to_end.py
```

The end-to-end test verifies:

- saved model loading;
- metadata/model feature agreement;
- dashboard dataset integrity;
- absence of target/future-outcome leakage in model inputs;
- reference prediction reproducibility;
- direct-model versus pipeline inference equivalence;
- VPS consistency;
- invalid Cascade ID handling;
- missing-feature handling;
- probability and VPS output bounds.

A successful run ends with:

```text
ALL END-TO-END TESTS PASSED
```

---

## 16. Running the Dashboard

Once the processed validation data and trained model are available:

```bash
streamlit run app.py
```

The application will start locally and provide the Trend Predictions, Trend Explorer and Feature Analysis views.

---

## 17. Experimental Outputs

Consolidated result tables are stored in:

```text
results/tables/
```

Final figures are stored in:

```text
results/figures/
```

These include:

- model-comparison results;
- ROC and Precision-Recall curves;
- confusion matrices;
- bootstrap confidence intervals;
- paired-bootstrap comparisons;
- feature-group ablation;
- permutation importance;
- partial-dependence analysis;
- network interaction analysis;
- held-out test evaluation.

---

## 18. Reproducibility

The project uses fixed random states for the established machine-learning experiments where applicable.

The final Random Forest uses:

```text
random_state = 42
```

The frozen training dataset recorded in the model metadata has SHA-256:

```text
9ba18fd054657adea3634a221445924cad569f3e138034c69c1404923ca87867
```

Additional dataset hashes are stored in:

```text
results/data_hashes.txt
```

These hashes can be used to verify that the frozen datasets have not changed between experimental runs.

---

## 19. Limitations

Several limitations should be considered when interpreting the prototype.

First, the model is evaluated on the dataset and trend-emergence definition used in this project. Performance should not automatically be assumed to generalise to other social-media platforms, time periods or definitions of virality.

Second, the target classes are imbalanced. Class weighting improves sensitivity to emerging cascades but introduces a precision-recall trade-off.

Third, the held-out test results are lower than some validation metrics, demonstrating the importance of evaluating the frozen model on unseen data.

Fourth, Random Forest probabilities are used to construct VPS, but VPS is not an independently calibrated or externally validated measure of real-world virality.

Fifth, feature-importance and partial-dependence analyses describe the fitted model's behaviour. They do not establish causal relationships between cascade characteristics and trend emergence.

Finally, the dashboard is a research prototype intended to demonstrate the predictive and analytical workflow. It is not a production social-media monitoring or forecasting system.

---

## 20. Project Status

The prototype currently includes:

- completed temporal and network feature modelling;
- baseline and model comparisons;
- class-imbalance analysis;
- feature ablation;
- statistical evaluation;
- held-out testing;
- explainability analysis;
- final model serialization;
- VPS inference pipeline;
- interactive Streamlit dashboard;
- end-to-end system testing.

The final experimental evidence, figures and tables are retained under the `results/` directory.