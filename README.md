# Hospital 30-Day Readmission Risk Prediction

An end-to-end machine learning project for predicting 30-day hospital readmission risk using structured hospital admission data, clinical-note representations, feature engineering, and a retrieval-based feature.


## Project Overview

Hospital readmissions can increase healthcare costs and create additional challenges for patients and healthcare providers. This project explores how machine learning can help identify patients who may be at higher risk of readmission within 30 days after discharge.

The project compares classical machine learning models with a hybrid Transformer model and investigates whether retrieving similar historical encounters can improve prediction performance.

The intended application is a clinical decision-support tool that helps healthcare teams prioritise patient follow-up. It does not replace clinical judgement or automatically make treatment decisions.

## Objectives

- Predict the 30-day readmission label as a binary classification task.
- Prepare and analyse linked hospital admission datasets.
- Engineer and select meaningful predictive features.
- Compare Logistic Regression, Random Forest, and a hybrid DistilBERT model.
- Evaluate a RAG-style retrieval feature based on similar historical encounters.
- Evaluate models using ROC-AUC, PR-AUC, precision, recall, and risk-ranking metrics.
- Investigate leakage prevention, temporal validation, calibration, and model limitations.

## Dataset

The primary dataset is the **India Hospital Readmission Dataset (2015–2024)**, sourced from Kaggle.

| Dataset characteristic | Value |
|---|---:|
| Original admissions | 120,000 |
| Distinct patients | 86,400 |
| Diagnosis records | 271,341 |
| Hospitals represented | 33 |
| Admissions retained after cleaning | 112,587 |
| Labelled readmission rate | Approximately 12.6% |

The dataset contains five linked tables. They are combined carefully to create one modelling row per admission and avoid duplicating admission records when multiple diagnoses are present.

### Data split

A chronological split is used to approximate prediction on future admissions.

| Split | Period | Admissions |
|---|---|---:|
| Training | 2015–2021 | 78,751 |
| Validation | 2022 | 11,404 |
| Testing | 2023–2024 | 22,432 |

The validation set supports model selection and development decisions. The test set is reserved for final evaluation.

## Methodology

### 1. Data preparation

- Load and inspect the linked dataset tables.
- Audit missing values, duplicates, feature ranges, and class balance.
- Prepare an admission-level dataset.
- Remove 7,413 in-hospital deaths from the modelling set.
- Exclude leakage-prone fields and information unavailable at prediction time.
- Construct a time-based training, validation, and test split.

### 2. Feature engineering and selection

The feature-engineering workflow explores additional predictive signals, including clinical flags, interactions, cost ratios, patient-history indicators, and leakage-aware target encodings.

Feature selection is evaluated using validation data to investigate whether a smaller set of features can retain predictive performance.

### 3. Classical machine learning

The project compares:

- **Logistic Regression:** an interpretable baseline for binary classification.
- **Random Forest:** an ensemble of decision trees.
- **HistGradientBoosting:** an additional tree-based benchmark.

### 4. Hybrid Transformer model

The project also evaluates a hybrid DistilBERT architecture that combines clinical-note text representations with structured patient features.

The Transformer workflow uses the Hugging Face Transformers library and PyTorch. GPU acceleration is used for computationally intensive experiments.

### 5. RAG-style retrieval feature

The retrieval experiment tests whether similar historical encounters can provide useful additional information.

1. Convert clinical notes into 384-dimensional embeddings using `all-MiniLM-L6-v2`.
2. Compare embeddings using cosine similarity.
3. Retrieve the top-k eligible historical encounters.
4. Calculate the readmission rate among the retrieved encounters.
5. Add that rate as an extra numerical feature.
6. Compare model performance with and without the retrieval feature.

The implementation applies time-aware eligibility rules: a historical encounter can be retrieved only after its 30-day outcome is observable, and other encounters from the same patient are excluded.

This is a retrieval-based feature-engineering experiment rather than a conventional generative question-answering system.

## Evaluation

The project uses several complementary metrics:

- **ROC-AUC:** measures ranking ability across classification thresholds.
- **PR-AUC:** evaluates the precision–recall trade-off, particularly useful for imbalanced labels.
- **Precision:** measures the proportion of flagged admissions labelled positive.
- **Recall:** measures the proportion of positive admissions identified.
- **Lift:** compares the positive rate in a selected high-risk group with the overall positive rate.
- **Brier score:** evaluates probabilistic predictions, with lower values indicating better probability accuracy.

The notebooks also explore patient-level bootstrap uncertainty, model calibration, operating points, and class-imbalance strategies.

## Key Results

The final presentation reports the following test ROC-AUC values:

| Model | Without retrieval | With retrieval |
|---|---:|---:|
| Logistic Regression | 0.7543 | 0.7543 |
| Random Forest | 0.7492 | 0.7482 |
| Hybrid DistilBERT | 0.7489 | 0.7490 |

The main finding is that adding the retrieval feature did not meaningfully improve performance in the reported experiment.

For Logistic Regression, the presentation reports the following result when reviewing the top 10% of admissions ranked by predicted risk:

- Precision: **40.9%**
- Recall: **32.4%**
- Lift: **3.3×**

These results describe performance on the available dataset and should not be interpreted as evidence of clinical effectiveness.

## Repository Structure

The following is a recommended structure for organising the uploaded notebooks. The filenames below are suggested clean names; rename the files in GitHub accordingly.

```text
hospital-readmission-risk-prediction/
│
├── README.md
│
├── notebooks/
│   ├── 01_readmission_final_pipeline_v3.ipynb
│   ├── 02_readmission_feature_engineering.ipynb
│   ├── 03_readmission_training_pipeline.ipynb
│   ├── 04_readmission_training_pipeline_v2.ipynb
│   ├── 05_quick_test_gpu.ipynb
│   ├── 06_full_run_gpu.ipynb
│   └── 07_clinicalbert_reference.ipynb
│
├── artifacts/
│   └── README.md
│
└── docs/
    └── README.md
```

The `artifacts/` and `docs/` folders are optional organisational placeholders, not claims that saved model files or additional documentation are already included.

## Notebook Guide

| Notebook | Purpose |
|---|---|
| `readmission_final_pipeline_v3.ipynb` | Main end-to-end pipeline, including data auditing, temporal validation, retrieval, class balancing, model evaluation, calibration experiments, and artifact saving. |
| `readmission_feature_engineering.ipynb` | Feature engineering, feature selection, model comparison, permutation importance, calibration, and operating-point analysis. |
| `readmission_training_pipeline.ipynb` | Full training workflow comparing classical and hybrid Transformer models, with and without retrieval. |
| `readmission_training_pipeline_v2.ipynb` | Additional version of the full training workflow for experimentation and comparison. |
| `quick_test_gpu.ipynb` | Short GPU run to check that the hybrid Transformer workflow executes successfully. |
| `full_run_gpu.ipynb` | Longer GPU experiment covering Logistic Regression, Random Forest, and hybrid DistilBERT, with and without retrieval, across multiple seeds. |
| `ClinicalBERT_Deep_Learning_Predicting_Hospital_Readmission.ipynb` | Separate ClinicalBERT-oriented deep-learning notebook for studying clinical-note-based readmission prediction. Its workflow should be treated separately from the main DistilBERT pipeline. |

**Recommended starting point:** Begin with `readmission_final_pipeline_v3.ipynb` for the most comprehensive documented workflow. The other notebooks provide alternative implementations and experiments; their results should not automatically be treated as identical to the final presentation results.

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Classical machine learning:** Scikit-learn
- **Text embeddings:** Sentence Transformers
- **Transformer models:** Hugging Face Transformers, PyTorch
- **Visualisation:** Matplotlib
- **Execution environment:** Google Colab, including T4 GPU experiments
- **Data source:** Kaggle

## How to Run

1. Clone or download this repository.
2. Open the `notebooks/` directory.
3. Upload the selected notebook to Google Colab or open it in a compatible Jupyter environment.
4. Configure access to the required Kaggle dataset and any necessary credentials.
5. Install the dependencies listed in the notebook.
6. Select a GPU runtime for the Transformer experiments when required.
7. Run the notebook cells in order.

The full training runs may require substantial compute time and memory. The quick-test notebook is intended for checking the GPU workflow before running the longer experiment.

Dataset downloads, runtime configuration, and model-artifact paths may need to be adjusted for your environment.

## Limitations and Responsible Use

This project is experimental and is not intended for direct clinical use.

A significant limitation is the quality of the readmission labels. In the available data, only approximately 1.2% of admissions labelled as readmitted have a later same-patient admission within 30 days. This raises concerns about label consistency and means the reported metrics cannot establish real-world hospital performance.

Additional work is needed in the following areas:

- Validate outcome labels against reliable clinical records.
- Perform external validation on separate hospital data.
- Evaluate calibration and decision thresholds before interpreting scores as probabilities.
- Complete fairness and subgroup evaluations.
- Implement and test privacy controls, role-based access, and audit logging.
- Validate the proposed worklist and clinical workflow with healthcare professionals.
- Monitor model drift and performance if a future deployment is undertaken.

The interface and hospital integrations are proposed concepts rather than a fully deployed clinical system.

## Future Improvements

- Validate the target labels using reliable clinical data.
- Test generalisation across hospitals and patient populations.
- Investigate whether better clinical notes improve retrieval and Transformer performance.
- Evaluate calibration and clinically appropriate thresholds.
- Complete fairness, privacy, and security assessments.
- Develop a monitored pilot with healthcare professionals.
- Reassess retrieval only if it demonstrates a measurable benefit.

## Conclusion

This project demonstrates an end-to-end approach to hospital readmission risk modelling, from data preparation and feature engineering to classical machine learning, Transformer experiments, retrieval-based feature augmentation, and time-aware evaluation.

The central finding is that the simpler Logistic Regression baseline remained competitive, while the retrieval feature did not provide a meaningful improvement on the current dataset.

The next step is not simply to add a more complex model, but to validate the data, establish reliable performance on real clinical records, and assess whether the system can support safe and useful follow-up decisions.

---

**Disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used as the sole basis for clinical decisions.
